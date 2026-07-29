#!/usr/bin/env bash
#
# Contention watchdog for benchmark runs on the shared GPU host.
#
# `benchmark/gpu_preflight.sh` establishes that the GPUs are free *before* a run starts. This
# watches whether they stay that way *during* it, and records host load alongside — a timing
# number from this machine is not reproducible without the load it was taken under.
#
#   log=$(mktemp); benchmark/gpu_watchdog.sh start --devices 0,1,2,3 --log "$log"
#   ... run the benchmark ...
#   benchmark/gpu_watchdog.sh stop  --log "$log"
#   benchmark/gpu_watchdog.sh check --log "$log"   # exit 1 if anyone else showed up
#
# Deliberately does not abort the run on contention. Our sections take minutes, and bailing out
# halfway would waste the GPU time we already took without giving the other job its device back
# any sooner. Instead we finish, then mark the results suspect so they get re-run rather than
# quietly believed.
#
# Exit codes for `check`:
#   0  clean      only our own processes touched the watched GPUs
#   1  contended  someone else appeared; treat the timings as invalid
#   3  environment
set -euo pipefail

CMD="${1:-}"
[[ -n "$CMD" ]] && shift || true

DEVICES=""
LOG=""
INTERVAL=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) DEVICES="$2"; shift 2 ;;
        --log) LOG="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        *) echo "WATCHDOG: fail — unknown argument '$1'" >&2; exit 3 ;;
    esac
done

[[ -n "$LOG" ]] || { echo "WATCHDOG: fail — --log is required" >&2; exit 3; }
PIDFILE="${LOG}.pid"
ME=$(id -un)

sample_once() {
    local now load
    now=$(date +%s)
    load=$(cut -d' ' -f1 /proc/loadavg)

    declare -A u2i
    while read -r i u; do
        u2i["$u"]="$i"
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' 'NF { print $1, $2 }')

    local any=0
    while read -r uuid pid mem; do
        local idx user
        idx="${u2i[$uuid]:-?}"
        user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
        [[ -n "$user" ]] || user="?"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$load" "$idx" "$pid" "$user" "$mem" >> "$LOG"
        any=1
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | awk -F', *' 'NF { print $1, $2, $3 }')

    # Always emit a row, even with no compute apps, so the load trace is continuous.
    if ((any == 0)); then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$now" "$load" "-" "-" "-" "-" >> "$LOG"
    fi
}

case "$CMD" in
    start)
        [[ -n "$DEVICES" ]] || { echo "WATCHDOG: fail — --devices is required for start" >&2; exit 3; }
        : > "$LOG"
        printf '# devices=%s owner=%s interval=%s\n' "$DEVICES" "$ME" "$INTERVAL" >> "$LOG"
        (
            while true; do
                sample_once
                sleep "$INTERVAL"
            done
        ) &
        echo $! > "$PIDFILE"
        echo "WATCHDOG: started (pid $(cat "$PIDFILE")), watching GPUs ${DEVICES}, log ${LOG}" >&2
        ;;

    stop)
        if [[ -f "$PIDFILE" ]]; then
            kill "$(cat "$PIDFILE")" 2>/dev/null || true
            rm -f "$PIDFILE"
            echo "WATCHDOG: stopped" >&2
        fi
        # One final reading, so the window closes on a real sample rather than mid-interval.
        sample_once
        ;;

    check)
        [[ -f "$LOG" ]] || { echo "WATCHDOG: fail — no log at ${LOG}" >&2; exit 3; }
        watched=$(head -1 "$LOG" | sed -n 's/.*devices=\([^ ]*\).*/\1/p')
        [[ -n "$DEVICES" ]] && watched="$DEVICES"

        # Load envelope over the run — the context every reported timing needs.
        awk -F'\t' '!/^#/ && $2 != "" { n++; s += $2; if ($2 > mx) mx = $2 }
            END { if (n) printf "WATCHDOG: load over run — mean %.1f, peak %.1f (%d samples)\n", s / n, mx, n }' \
            "$LOG" >&2

        foreign=$(awk -F'\t' -v me="$ME" -v dev="$watched" '
            BEGIN { split(dev, a, ","); for (k in a) want[a[k]] = 1 }
            !/^#/ && $3 != "-" && ($3 in want) && $5 != me { print $3, $4, $5 }
        ' "$LOG" | sort -u)

        if [[ -n "$foreign" ]]; then
            echo "WATCHDOG: contended — other users' processes appeared on the watched GPUs:" >&2
            echo "$foreign" | awk '{ printf "  GPU %s  pid %s  user %s\n", $1, $2, $3 }' >&2
            echo "WATCHDOG: timings from this run are not trustworthy; re-run when the host is quiet." >&2
            exit 1
        fi
        echo "WATCHDOG: clean — no foreign processes on GPUs ${watched} for the whole run" >&2
        ;;

    *)
        echo "WATCHDOG: fail — expected one of: start | stop | check" >&2
        exit 3
        ;;
esac
