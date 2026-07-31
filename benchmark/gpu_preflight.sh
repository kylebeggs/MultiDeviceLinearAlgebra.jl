#!/usr/bin/env bash
#
# GPU availability guard. The GPU host is *shared* — other people's jobs live on it — so nothing
# in benchmark/ may assume it owns the machine. This script answers one question: "are there N
# GPUs genuinely free right now, and if so which N should I take?"
#
# Used by both the manual `benchmark/gpu.jl` run and the CI GPU jobs, which is why it lives in the
# repo rather than in someone's shell history.
#
#   eval "$(benchmark/gpu_preflight.sh --need 4 --max-load-frac 0.6)"   # exports CUDA_VISIBLE_DEVICES
#   benchmark/gpu_preflight.sh --need 2 --bare                          # prints "0,1"
#   benchmark/gpu_preflight.sh --need 2 --wait 7200 --prefer-far --bare # wait up to 2h, worst link
#
# Options:
#   --need N            GPUs required (default 4)
#   --samples N         readings per probe (default 3)
#   --interval SEC      gap between readings (default 5)
#   --max-util PCT      a GPU above this is not free (default 5)
#   --max-mem MIB       a GPU above this is not free (default 200)
#   --max-load-frac F   also require host load/cores below F (unset = no load gate)
#   --wait SEC          keep re-probing until free, up to SEC (default 0 = probe once)
#   --poll SEC          gap between probes while waiting (default 120)
#   --wait-hook CMD     run each wait iteration; a nonzero exit abandons the wait (code 4)
#   --prefer-far        pick the WORST-connected free subset instead of the best
#   --bare              print just the device list, not an export statement
#
# Exit codes are distinct so a caller can report *why* it stood down:
#   0  ok            selection printed on stdout
#   1  busy          fewer than --need GPUs free
#   2  loaded        host load above --max-load-frac
#   3  environment   nvidia-smi missing, bad arguments, more GPUs asked for than exist
#   4  abandoned     --wait-hook said to stop waiting (e.g. a newer commit superseded this run)
#
# A one-line summary always goes to stderr, success or failure, so callers can log the context a
# timing number is meaningless without.
#
# Waiting holds no device. It only re-reads nvidia-smi, so a job that waits two hours costs another
# user nothing — which is the whole point: standing down immediately means a pull request simply
# never gets GPU coverage, while grabbing a busy GPU is not an option at all.
set -euo pipefail

NEED=4
SAMPLES=3
INTERVAL=5
MAX_UTIL=5      # percent
MAX_MEM=200     # MiB; the driver alone sits at ~4 MiB on an idle A30
MAX_LOAD_FRAC=""
WAIT=0
POLL=120
WAIT_HOOK=""
PREFER_FAR=0
BARE=0

usage() {
    sed -n '3,/^set -euo pipefail/p' "$0" | sed '$d; s/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --need) NEED="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --max-util) MAX_UTIL="$2"; shift 2 ;;
        --max-mem) MAX_MEM="$2"; shift 2 ;;
        --max-load-frac) MAX_LOAD_FRAC="$2"; shift 2 ;;
        --wait) WAIT="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        --wait-hook) WAIT_HOOK="$2"; shift 2 ;;
        --prefer-far) PREFER_FAR=1; shift ;;
        --bare) BARE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "PREFLIGHT: fail — unknown argument '$1'" >&2; exit 3 ;;
    esac
done

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "PREFLIGHT: fail — nvidia-smi not found" >&2
    exit 3
}

# Device inventory is fixed for the life of the host, so read it once, outside the wait loop. This
# also makes an impossible request fail immediately instead of burning the whole --wait window on a
# condition that can never become true.
mapfile -t IDX < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
NGPU=${#IDX[@]}
NCPU=$(nproc)

if ((NEED > NGPU)); then
    echo "PREFLIGHT: fail — asked for ${NEED} GPUs, host has ${NGPU}" >&2
    exit 3
fi

# ── One availability probe ───────────────────────────────────────────────────
# Sets FREE and SUMMARY; returns 0 ok / 1 busy / 2 loaded. Everything it reads is transient, so the
# wait loop below re-runs the whole thing rather than caching any of it.
probe() {
    local load1 load_frac i u util mem idx _pid s

    load1=$(cut -d' ' -f1 /proc/loadavg)
    load_frac=$(awk -v l="$load1" -v n="$NCPU" 'BEGIN { printf "%.3f", l / n }')

    # ── Host load ────────────────────────────────────────────────────────────
    # Only gated when the caller asks. Correctness tests do not care how noisy the host is; timing
    # runs very much do, because with P2P degraded the ghost exchange stages through host memory
    # and lands squarely on the contended CPU.
    if [[ -n "$MAX_LOAD_FRAC" ]]; then
        if awk -v a="$load_frac" -v b="$MAX_LOAD_FRAC" 'BEGIN { exit !(a > b) }'; then
            SUMMARY="load ${load1} over ${NCPU} cores = ${load_frac} > ${MAX_LOAD_FRAC}"
            return 2
        fi
    fi

    # ── Sample the GPUs over a window ────────────────────────────────────────
    # A single instantaneous reading is not enough: a busy job reads 0% utilisation in the gap
    # between kernel launches. Take the worst value each GPU shows across the whole window.
    local -A MAXUTIL MAXMEM HASAPP UUID2IDX
    for i in "${IDX[@]}"; do
        MAXUTIL[$i]=0
        MAXMEM[$i]=0
        HASAPP[$i]=0
    done

    while read -r i u; do
        UUID2IDX["$u"]="$i"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' 'NF { print $1, $2 }')

    # NOTE the `$` in every subscript below, including inside (( )). For an ASSOCIATIVE array bash
    # treats a subscript as a literal string, not an arithmetic expression — so `MAXUTIL[i]` reads
    # the key "i", which is unset, i.e. 0. Written that way this whole guard silently reported every
    # GPU free no matter how busy the host was, which is the one failure it exists to prevent.
    for ((s = 1; s <= SAMPLES; s++)); do
        while read -r i util mem; do
            if ((util > MAXUTIL[$i])); then MAXUTIL[$i]=$util; fi
            if ((mem > MAXMEM[$i])); then MAXMEM[$i]=$mem; fi
        done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
            --format=csv,noheader,nounits | awk -F', *' 'NF { print $1, $2, $3 }')

        # A resident compute process disqualifies a GPU outright, even at 0% utilisation — someone
        # between phases of their job still owns that device.
        while read -r u _pid; do
            idx="${UUID2IDX[$u]:-}"
            if [[ -n "$idx" ]]; then HASAPP[$idx]=1; fi
        done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid \
            --format=csv,noheader 2>/dev/null | awk -F', *' 'NF { print $1, $2 }')

        if ((s < SAMPLES)); then sleep "$INTERVAL"; fi
    done

    FREE=()
    for i in "${IDX[@]}"; do
        if ((MAXUTIL[$i] <= MAX_UTIL)) && ((MAXMEM[$i] <= MAX_MEM)) && ((HASAPP[$i] == 0)); then
            FREE+=("$i")
        fi
    done

    SUMMARY="${#FREE[@]}/${NGPU} GPUs free (need ${NEED}), load ${load1}/${NCPU} cores = ${load_frac}"
    ((${#FREE[@]} >= NEED)) || return 1
    return 0
}

# ── Wait for the host to have room ───────────────────────────────────────────
# `busy` and `loaded` are both transient, so both are worth waiting out. With --wait 0 this loop
# runs exactly once and the script behaves as it always did.
DEADLINE=$(($(date +%s) + WAIT))
RC=0
while :; do
    RC=0
    probe || RC=$?
    if ((RC == 0)); then break; fi

    NOW=$(date +%s)
    if ((NOW + POLL > DEADLINE)); then break; fi

    # The hook is how a caller abandons a wait that has stopped being worth finishing — CI passes
    # one that compares the pull request's live head against the commit this run checked out, so a
    # superseded run yields the host to the newer one instead of sitting on a two-hour timer.
    if [[ -n "$WAIT_HOOK" ]] && ! eval "$WAIT_HOOK"; then
        echo "PREFLIGHT: abandoned — wait hook asked to stop; ${SUMMARY}" >&2
        exit 4
    fi

    case "$RC" in
        1) REASON="busy" ;;
        2) REASON="loaded" ;;
        *) REASON="unavailable" ;;
    esac
    echo "PREFLIGHT: ${REASON}, waiting ${POLL}s ($((DEADLINE - NOW))s left) — ${SUMMARY}" >&2
    sleep "$POLL"
done

if ((RC == 2)); then
    echo "PREFLIGHT: loaded — ${SUMMARY}" >&2
    exit 2
fi
if ((RC != 0)); then
    echo "PREFLIGHT: busy — ${SUMMARY}" >&2
    exit 1
fi

# ── Pick the subset ──────────────────────────────────────────────────────────
# Which GPUs we take matters, not just how many. Read the real topology rather than hardcoding
# pairs, then grow the selection greedily. Ties go to the lowest index, so the result is
# deterministic.
#
# The default is the BEST-connected subset: an accidental cross-socket set makes the halo exchange
# look far worse than the hardware can do, which is wrong for a benchmark. --prefer-far inverts it
# for correctness runs, which want the opposite — the SYS-class pairs are the ones a mistranslating
# IOMMU corrupts hardest, and the ones `_p2p_copy_ok` and its host-staging fallback exist to
# protect. Picking well-connected devices there means test_cross_socket.jl silently degrades to
# testing an adjacent pair.
declare -A LINK
if topo=$(nvidia-smi topo -m 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'); then
    while read -r line; do
        [[ "$line" =~ ^GPU([0-9]+) ]] || continue
        src="${BASH_REMATCH[1]}"
        read -ra fields <<< "$line"
        for ((j = 0; j < NGPU; j++)); do
            LINK["$src,$j"]="${fields[$((j + 1))]:-SYS}"
        done
    done <<< "$topo"
fi

link_rank() {
    case "$1" in
        NV*) echo 0 ;;   # NVLink
        PIX) echo 1 ;;   # same PCIe switch
        PXB) echo 2 ;;
        PHB) echo 3 ;;
        NODE) echo 4 ;;  # same NUMA node, across host bridges
        SYS) echo 5 ;;   # across sockets
        *) echo 9 ;;
    esac
}

declare -A CHOSEN
SEL=("${FREE[0]}")
CHOSEN[${FREE[0]}]=1
while ((${#SEL[@]} < NEED)); do
    best=""
    if ((PREFER_FAR == 1)); then bestscore=-1; else bestscore=999999; fi
    for c in "${FREE[@]}"; do
        [[ -n "${CHOSEN[$c]:-}" ]] && continue
        score=0
        for s in "${SEL[@]}"; do
            score=$((score + $(link_rank "${LINK["$s,$c"]:-SYS}")))
        done
        if ((PREFER_FAR == 1)); then
            if ((score > bestscore)); then bestscore=$score; best="$c"; fi
        else
            if ((score < bestscore)); then bestscore=$score; best="$c"; fi
        fi
    done
    SEL+=("$best")
    CHOSEN[$best]=1
done

mapfile -t SORTED < <(printf '%s\n' "${SEL[@]}" | sort -n)
LIST=$(IFS=,; echo "${SORTED[*]}")

if ((PREFER_FAR == 1)); then WHICH="worst-connected"; else WHICH="best-connected"; fi
echo "PREFLIGHT: ok — taking ${WHICH} GPUs ${LIST}; ${SUMMARY}" >&2
if ((BARE == 1)); then
    echo "$LIST"
else
    echo "export CUDA_VISIBLE_DEVICES=${LIST}"
fi
