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
#
# Exit codes are distinct so a caller can report *why* it stood down:
#   0  ok            selection printed on stdout
#   1  busy          fewer than --need GPUs free
#   2  loaded        host load above --max-load-frac
#   3  environment   nvidia-smi missing, bad arguments
#
# A one-line summary always goes to stderr, success or failure, so callers can log the context a
# timing number is meaningless without.
set -euo pipefail

NEED=4
SAMPLES=3
INTERVAL=5
MAX_UTIL=5      # percent
MAX_MEM=200     # MiB; the driver alone sits at ~4 MiB on an idle A30
MAX_LOAD_FRAC=""
BARE=0

usage() {
    sed -n '3,20p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --need) NEED="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --max-util) MAX_UTIL="$2"; shift 2 ;;
        --max-mem) MAX_MEM="$2"; shift 2 ;;
        --max-load-frac) MAX_LOAD_FRAC="$2"; shift 2 ;;
        --bare) BARE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "PREFLIGHT: fail — unknown argument '$1'" >&2; exit 3 ;;
    esac
done

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "PREFLIGHT: fail — nvidia-smi not found" >&2
    exit 3
}

NCPU=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
LOAD_FRAC=$(awk -v l="$LOAD1" -v n="$NCPU" 'BEGIN { printf "%.3f", l / n }')

# ── Host load ────────────────────────────────────────────────────────────────
# Only gated when the caller asks. Correctness tests do not care how noisy the host is; timing
# runs very much do, because with P2P degraded the ghost exchange stages through host memory and
# lands squarely on the contended CPU.
if [[ -n "$MAX_LOAD_FRAC" ]]; then
    if awk -v a="$LOAD_FRAC" -v b="$MAX_LOAD_FRAC" 'BEGIN { exit !(a > b) }'; then
        echo "PREFLIGHT: loaded — load ${LOAD1} over ${NCPU} cores = ${LOAD_FRAC} > ${MAX_LOAD_FRAC}" >&2
        exit 2
    fi
fi

# ── Sample the GPUs over a window ────────────────────────────────────────────
# A single instantaneous reading is not enough: a busy job reads 0% utilisation in the gap between
# kernel launches. Take the worst value each GPU shows across the whole window.
mapfile -t IDX < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits)
NGPU=${#IDX[@]}

declare -A MAXUTIL MAXMEM HASAPP UUID2IDX
for i in "${IDX[@]}"; do
    MAXUTIL[$i]=0
    MAXMEM[$i]=0
    HASAPP[$i]=0
done

while read -r i u; do
    UUID2IDX["$u"]="$i"
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' 'NF { print $1, $2 }')

for ((s = 1; s <= SAMPLES; s++)); do
    while read -r i util mem; do
        if ((util > MAXUTIL[i])); then MAXUTIL[$i]=$util; fi
        if ((mem > MAXMEM[i])); then MAXMEM[$i]=$mem; fi
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
    if ((MAXUTIL[i] <= MAX_UTIL)) && ((MAXMEM[i] <= MAX_MEM)) && ((HASAPP[i] == 0)); then
        FREE+=("$i")
    fi
done

SUMMARY="${#FREE[@]}/${NGPU} GPUs free (need ${NEED}), load ${LOAD1}/${NCPU} cores = ${LOAD_FRAC}"

if ((NEED > NGPU)); then
    echo "PREFLIGHT: fail — asked for ${NEED} GPUs, host has ${NGPU}" >&2
    exit 3
fi
if ((${#FREE[@]} < NEED)); then
    echo "PREFLIGHT: busy — ${SUMMARY}" >&2
    exit 1
fi

# ── Pick the best-connected subset ───────────────────────────────────────────
# Which GPUs we take matters, not just how many: an accidental cross-socket set makes the halo
# exchange look far worse than the hardware can do. Read the real topology rather than hardcoding
# pairs, then grow the selection greedily, always adding whichever free GPU has the best links to
# what is already chosen. Ties go to the lowest index, so the result is deterministic.
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
    bestscore=999999
    for c in "${FREE[@]}"; do
        [[ -n "${CHOSEN[$c]:-}" ]] && continue
        score=0
        for s in "${SEL[@]}"; do
            score=$((score + $(link_rank "${LINK["$s,$c"]:-SYS}")))
        done
        if ((score < bestscore)); then
            bestscore=$score
            best="$c"
        fi
    done
    SEL+=("$best")
    CHOSEN[$best]=1
done

mapfile -t SORTED < <(printf '%s\n' "${SEL[@]}" | sort -n)
LIST=$(IFS=,; echo "${SORTED[*]}")

echo "PREFLIGHT: ok — taking GPUs ${LIST}; ${SUMMARY}" >&2
if ((BARE == 1)); then
    echo "$LIST"
else
    echo "export CUDA_VISIBLE_DEVICES=${LIST}"
fi
