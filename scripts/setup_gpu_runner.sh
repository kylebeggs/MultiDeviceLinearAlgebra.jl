#!/usr/bin/env bash
#
# Install the self-hosted GitHub Actions runner on the shared GPU host.
#
# Run this ON THE HOST, as a user with sudo. It is the scripted form of the runbook in
# docs/gpu-ci-runner.md — kept as a committed script rather than copy-paste blocks so it is
# reviewable, re-runnable, and survives a repository transfer (change --repo, nothing else).
#
# Generate the registration token first, on a machine where `gh` is authenticated. It is valid for
# one hour and is NOT a long-lived credential:
#
#   gh api -X POST repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners/registration-token --jq .token
#
# Then, on the host:
#
#   ./scripts/setup_gpu_runner.sh --token <TOKEN>
#
# Options:
#   --token TOK        registration token (or set RUNNER_TOKEN)
#   --repo OWNER/NAME  default kylebeggs/MultiDeviceLinearAlgebra.jl
#   --name NAME        runner name, default sasquatch
#   --user USER        unprivileged account to create and run as, default ghrunner
#   --labels LIST      default self-hosted,linux,x64,cuda,gpu,sasquatch
#   --version VER      pin a runner version (default: resolve the latest release)
#   --skip-checksum    proceed when the published SHA256 cannot be found (NOT recommended)
#   --uninstall        stop, uninstall and de-register; leaves the account in place
#
# The `cuda` label is what `runs-on: [self-hosted, cuda]` in .github/workflows/BenchmarkGPU.yml
# matches. Keep it.
#
# Security: the account is created WITHOUT sudo, docker, or any privileged group membership, and
# holds no secrets. CUDA needs no group membership — /dev/nvidia* is world-accessible. This limits
# the blast radius; the controls that actually keep fork code off the box are the repository's
# "require approval for all external contributors" setting and the same-repo `if:` gate on every
# job. See docs/gpu-ci-runner.md.
set -euo pipefail

REPO="kylebeggs/MultiDeviceLinearAlgebra.jl"
RUNNER_NAME="sasquatch"
RUNNER_USER="ghrunner"
LABELS="self-hosted,linux,x64,cuda,gpu,sasquatch"
VERSION=""
TOKEN="${RUNNER_TOKEN:-}"
SKIP_CHECKSUM=0
UNINSTALL=0

usage() { sed -n '3,/^set -euo pipefail/p' "$0" | sed '$d; s/^# \{0,1\}//'; }
say() { echo "==> $*" >&2; }
die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --token) TOKEN="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        --name) RUNNER_NAME="$2"; shift 2 ;;
        --user) RUNNER_USER="$2"; shift 2 ;;
        --labels) LABELS="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --skip-checksum) SKIP_CHECKSUM=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument '$1'" ;;
    esac
done

HOME_DIR="/home/${RUNNER_USER}"
RUNNER_DIR="${HOME_DIR}/actions-runner"

# ── Uninstall ────────────────────────────────────────────────────────────────
if ((UNINSTALL == 1)); then
    [[ -n "$TOKEN" ]] || die "--uninstall needs a REMOVE token: gh api -X POST repos/${REPO}/actions/runners/remove-token --jq .token"
    say "Stopping and uninstalling the service"
    sudo "${RUNNER_DIR}/svc.sh" stop || true
    sudo "${RUNNER_DIR}/svc.sh" uninstall || true
    say "De-registering the runner"
    sudo -u "$RUNNER_USER" bash -c "cd '${RUNNER_DIR}' && ./config.sh remove --token '${TOKEN}'"
    say "Done. The ${RUNNER_USER} account was left in place; remove it with:"
    echo "    sudo deluser --remove-home ${RUNNER_USER}" >&2
    exit 0
fi

[[ -n "$TOKEN" ]] || die "no registration token. Pass --token or set RUNNER_TOKEN. Generate with:
    gh api -X POST repos/${REPO}/actions/runners/registration-token --jq .token"

# ── Preconditions ────────────────────────────────────────────────────────────
[[ "$(uname -s)" == "Linux" ]] || die "this installs the linux-x64 runner; got $(uname -s)"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found — is this the GPU host?"
command -v curl >/dev/null 2>&1 || die "curl not found"
sudo -v || die "this script needs sudo for the account and the systemd service"

say "GPUs visible on this host:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader >&2

if [[ -d "$RUNNER_DIR" ]]; then
    die "${RUNNER_DIR} already exists. Re-run with --uninstall first, or remove it by hand."
fi

# ── 1. Unprivileged account ──────────────────────────────────────────────────
if id -u "$RUNNER_USER" >/dev/null 2>&1; then
    say "Account ${RUNNER_USER} already exists — leaving it alone"
else
    say "Creating unprivileged account ${RUNNER_USER} (no sudo, no extra groups)"
    sudo adduser --disabled-password --gecos "GitHub Actions runner" "$RUNNER_USER"
fi

say "Checking ${RUNNER_USER} can see the GPUs"
sudo -u "$RUNNER_USER" nvidia-smi -L >&2 || die "${RUNNER_USER} cannot reach the GPUs"

# ── 2. Resolve and download the runner ───────────────────────────────────────
# Not pinned by default: the release moves, persistent runners self-update anyway, and a stale pin
# in a doc is how you end up installing a version that no longer registers.
RELEASE_JSON=$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest) \
    || die "could not reach the actions/runner releases API"

if [[ -z "$VERSION" ]]; then
    VERSION=$(printf '%s' "$RELEASE_JSON" | grep -o '"tag_name"[[:space:]]*:[[:space:]]*"v[^"]*"' | head -1 | sed 's/.*"v\([^"]*\)"/\1/')
    [[ -n "$VERSION" ]] || die "could not determine the latest runner version"
fi
say "Installing actions-runner v${VERSION}"

TARBALL="actions-runner-linux-x64-${VERSION}.tar.gz"
URL="https://github.com/actions/runner/releases/download/v${VERSION}/${TARBALL}"

# The release body carries the published digests between literal markers. If that format ever
# changes we stop rather than install an unverified binary onto a machine other people use.
SHA=$(printf '%s' "$RELEASE_JSON" | grep -o "BEGIN SHA linux-x64 -->[0-9a-f]\{64\}" | head -1 | grep -o '[0-9a-f]\{64\}' || true)
if [[ -z "$SHA" ]]; then
    if ((SKIP_CHECKSUM == 1)); then
        say "WARNING: no published SHA256 found; continuing because --skip-checksum was given"
    else
        die "could not find the published SHA256 for v${VERSION} in the release body.
Verify it by hand at https://github.com/actions/runner/releases/tag/v${VERSION}, then re-run with
--version ${VERSION} --skip-checksum, or pin a version whose digest you have checked."
    fi
fi

say "Downloading ${TARBALL}"
sudo -u "$RUNNER_USER" mkdir -p "$RUNNER_DIR"
sudo -u "$RUNNER_USER" curl -fsSL -o "${RUNNER_DIR}/${TARBALL}" "$URL"

if [[ -n "$SHA" ]]; then
    say "Verifying SHA256"
    echo "${SHA}  ${RUNNER_DIR}/${TARBALL}" | sha256sum -c - >&2 \
        || die "checksum mismatch — do NOT install this. Delete ${RUNNER_DIR} and investigate."
fi

sudo -u "$RUNNER_USER" tar xzf "${RUNNER_DIR}/${TARBALL}" -C "$RUNNER_DIR"
sudo -u "$RUNNER_USER" rm -f "${RUNNER_DIR}/${TARBALL}"

# .NET runtime prerequisites (libicu and friends). Ships with the runner, needs root.
if [[ -x "${RUNNER_DIR}/bin/installdependencies.sh" ]]; then
    say "Installing runner OS dependencies"
    sudo "${RUNNER_DIR}/bin/installdependencies.sh" >&2 || say "WARNING: installdependencies.sh failed; continuing (they may already be present)"
fi

# ── 3. Register ──────────────────────────────────────────────────────────────
say "Registering as '${RUNNER_NAME}' with labels: ${LABELS}"
sudo -u "$RUNNER_USER" bash -c "cd '${RUNNER_DIR}' && ./config.sh \
    --url 'https://github.com/${REPO}' \
    --token '${TOKEN}' \
    --name '${RUNNER_NAME}' \
    --labels '${LABELS}' \
    --work _work \
    --unattended --replace"

# ── 4. Service ───────────────────────────────────────────────────────────────
say "Installing and starting the systemd service"
sudo "${RUNNER_DIR}/svc.sh" install "$RUNNER_USER"
sudo "${RUNNER_DIR}/svc.sh" start
sudo "${RUNNER_DIR}/svc.sh" status >&2

# ── 5. Verify ────────────────────────────────────────────────────────────────
cat >&2 <<EOF

==> Installed. Confirm GitHub agrees, from a machine with gh authenticated:

    gh api repos/${REPO}/actions/runners \\
      --jq '.runners[] | {name, status, labels: [.labels[].name]}'

Expect "status": "online" with the 'cuda' label present.

Then sanity-check the preflight guard against this host's real topology:

    ./benchmark/gpu_preflight.sh --need 2 --prefer-far --bare   # should cross a NUMA node
    ./benchmark/gpu_preflight.sh --need 4 --bare                # should stay on one socket

To remove everything later:

    ./scripts/setup_gpu_runner.sh --uninstall --token "\$(gh api -X POST \\
      repos/${REPO}/actions/runners/remove-token --jq .token)"
EOF
