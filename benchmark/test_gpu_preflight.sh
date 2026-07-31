#!/usr/bin/env bash
#
# Tests for benchmark/gpu_preflight.sh, driven by a mock nvidia-smi on PATH.
#
#   benchmark/test_gpu_preflight.sh
#
# Needs bash 4+ (the script under test uses `mapfile` and associative arrays), so it runs on the
# GPU host or any Linux box — NOT on stock macOS, whose /bin/bash is 3.2.
#
# The mock host is 4 GPUs on two sockets: 0,1 on socket 0 (PIX between them), 2,3 on socket 1, and
# every crossing pair SYS. That is the shape that matters — the selection logic has to be able to
# choose *against* connectivity for correctness runs, which is the whole point of --prefer-far.
#
# /proc/loadavg is not mocked; the load-gate cases use thresholds that any real load either always
# or never exceeds, so they stay deterministic without pretending to own /proc.
set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gpu_preflight.sh"
[[ -x "$SCRIPT" ]] || { echo "not found or not executable: $SCRIPT" >&2; exit 3; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export MOCK_STATE="$TMP/state"

# ── Mock nvidia-smi ──────────────────────────────────────────────────────────
# MOCK_BUSY        comma-separated indices that report load and a resident compute process
# MOCK_FREE_AFTER  after this many utilisation queries, everything reports free (0 = never)
cat > "$TMP/nvidia-smi" <<'MOCK'
#!/usr/bin/env bash
busy=",${MOCK_BUSY:-},"
free_after="${MOCK_FREE_AFTER:-0}"
args="$*"

if [[ "$args" == *"topo -m"* ]]; then
    printf '\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity\n'
    printf 'GPU0\t X \tPIX\tSYS\tSYS\t0-15\t0\n'
    printf 'GPU1\tPIX\t X \tSYS\tSYS\t0-15\t0\n'
    printf 'GPU2\tSYS\tSYS\t X \tPIX\t16-31\t1\n'
    printf 'GPU3\tSYS\tSYS\tPIX\t X \t16-31\t1\n'
    exit 0
fi

if [[ "$args" == *"--query-gpu=index,uuid"* ]]; then
    for i in 0 1 2 3; do echo "$i, GPU-uuid-$i"; done
    exit 0
fi

# Only this branch advances the clock, so a probe's utilisation and compute-app views always agree.
if [[ "$args" == *"--query-gpu=index,utilization.gpu,memory.used"* ]]; then
    n=0
    [[ -f "$MOCK_STATE" ]] && n=$(cat "$MOCK_STATE")
    n=$((n + 1)); echo "$n" > "$MOCK_STATE"
    relented=0
    if ((free_after > 0)) && ((n > free_after)); then relented=1; fi
    for i in 0 1 2 3; do
        if [[ "$busy" == *",$i,"* ]] && ((relented == 0)); then
            echo "$i, 95, 8000"
        else
            echo "$i, 0, 4"
        fi
    done
    exit 0
fi

if [[ "$args" == *"--query-compute-apps"* ]]; then
    n=0
    [[ -f "$MOCK_STATE" ]] && n=$(cat "$MOCK_STATE")
    relented=0
    if ((free_after > 0)) && ((n > free_after)); then relented=1; fi
    if ((relented == 0)); then
        IFS=',' read -ra b <<< "${MOCK_BUSY:-}"
        for i in "${b[@]}"; do [[ -n "$i" ]] && echo "GPU-uuid-$i, 4242"; done
    fi
    exit 0
fi

if [[ "$args" == *"--query-gpu=index"* ]]; then
    for i in 0 1 2 3; do echo "$i"; done
    exit 0
fi
exit 0
MOCK
chmod +x "$TMP/nvidia-smi"
export PATH="$TMP:$PATH"

PASS=0
FAIL=0

# Set the mock host's state. A helper rather than a `VAR=x run ...` prefix on purpose: bash keeps
# such assignments in scope after a *function* returns, so MOCK_FREE_AFTER from one case would
# quietly leak into the next and make a "still busy" test pass for the wrong reason.
mock() {
    export MOCK_BUSY="$1"
    export MOCK_FREE_AFTER="$2"
    rm -f "$MOCK_STATE"
}

# run <expected-rc> <expected-stdout-or-empty> <description> -- <args...>
run() {
    local want_rc="$1" want_out="$2" desc="$3"; shift 4
    local out rc
    out=$("$SCRIPT" "$@" 2>"$TMP/err"); rc=$?
    if [[ "$rc" != "$want_rc" ]]; then
        echo "FAIL  $desc"
        echo "        expected exit $want_rc, got $rc"
        echo "        stderr: $(tail -n2 "$TMP/err")"
        FAIL=$((FAIL + 1)); return
    fi
    if [[ -n "$want_out" && "$out" != "$want_out" ]]; then
        echo "FAIL  $desc"
        echo "        expected stdout '$want_out', got '$out'"
        FAIL=$((FAIL + 1)); return
    fi
    echo "ok    $desc"
    PASS=$((PASS + 1))
}

FAST=(--samples 1 --interval 0)

echo "── availability ──"
mock "" 0
run 0 "0,1" "all free, 2 needed → a pair" -- "${FAST[@]}" --need 2 --bare
mock "" 0
run 0 "0,1,2,3" "all free, 4 needed → every device" -- "${FAST[@]}" --need 4 --bare
mock "0,1,2" 0
run 1 "" "3 of 4 busy, 2 needed → busy" -- "${FAST[@]}" --need 2 --bare
mock "0" 0
run 0 "1,2" "a busy GPU is never selected" -- "${FAST[@]}" --need 2 --prefer-far --bare

echo "── topology-aware selection ──"
mock "" 0
run 0 "0,1" "default takes the best-connected pair (PIX, same socket)" -- \
    "${FAST[@]}" --need 2 --bare
mock "" 0
run 0 "0,2" "--prefer-far takes the worst-connected pair (SYS, cross-socket)" -- \
    "${FAST[@]}" --need 2 --prefer-far --bare

echo "── fail fast, never wait on a permanent condition ──"
# If this ever starts waiting it will hang the suite rather than fail, which is the point: asking
# for more GPUs than exist can never become true, so --wait must not apply to it.
mock "" 0
run 3 "" "--need above the device count → environment error, no wait" -- \
    "${FAST[@]}" --need 99 --wait 600 --poll 1 --bare
mock "" 0
run 3 "" "unknown argument → environment error" -- --nonsense

echo "── load gate ──"
# -1 rather than 0: load is always >= 0, so this trips even on a perfectly idle host.
mock "" 0
run 2 "" "--max-load-frac -1 → loaded" -- "${FAST[@]}" --need 2 --max-load-frac -1 --bare
mock "" 0
run 0 "0,1" "--max-load-frac 999 → gate never trips" -- \
    "${FAST[@]}" --need 2 --max-load-frac 999 --bare

echo "── waiting ──"
mock "0,1,2" 2
run 0 "0,1" "waits, then succeeds once the host frees up" -- \
    "${FAST[@]}" --need 2 --wait 60 --poll 1 --bare
mock "0,1,2" 0
run 1 "" "wait window expires while still busy → busy" -- \
    "${FAST[@]}" --need 2 --wait 2 --poll 1 --bare
mock "0,1,2" 0
run 4 "" "--wait-hook refusing → abandoned (4), distinct from busy (1)" -- \
    "${FAST[@]}" --need 2 --wait 60 --poll 1 --wait-hook false --bare
mock "0,1,2" 2
run 0 "0,1" "--wait-hook agreeing does not interrupt the wait" -- \
    "${FAST[@]}" --need 2 --wait 60 --poll 1 --wait-hook true --bare
mock "0,1,2" 0
run 2 "" "a loaded host is waited out too, and still reports loaded" -- \
    "${FAST[@]}" --need 2 --max-load-frac -1 --wait 2 --poll 1 --bare

echo "── output shape ──"
mock "" 0
run 0 "export CUDA_VISIBLE_DEVICES=0,1" "default output is an eval-able export" -- \
    "${FAST[@]}" --need 2

echo
echo "passed $PASS, failed $FAIL"
((FAIL == 0))
