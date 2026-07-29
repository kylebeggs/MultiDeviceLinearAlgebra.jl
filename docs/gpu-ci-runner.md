# Self-hosted GPU runner — setup and security model

CI on GitHub-hosted runners has no GPU, so the device half of this package has never been tested or
benchmarked automatically. `benchmark/benchmarks.jl` covers only the host-side construction path,
which means a green benchmark table on a pull request says nothing at all about device performance
— a trap that has already caused one PR's results to be misread. This runner closes that gap.

The host is **shared with other researchers**. Everything below follows from that.

## Security model

This repository is **public**. Without gating, anyone could fork it, edit a file the workflow
executes, open a pull request, and run arbitrary code on the GPU box. Fork PRs cannot modify the
workflow file itself — that is always read from the base branch — but they fully control the
sources it runs and the packages it instantiates.

Three controls, all required:

| # | Control | Where |
|---|---|---|
| 1 | **Require approval for all external contributors** | Repo → Settings → Actions → General → Fork pull request workflows |
| 2 | **Same-repo job gate** — fork PRs never queue | `if:` on every job in `.github/workflows/BenchmarkGPU.yml` |
| 3 | **Dedicated unprivileged user** — runner cannot `sudo`, holds no secrets | `ghrunner` account on the host |

Control 1 is already applied:

```console
$ gh api repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/permissions/fork-pr-contributor-approval
{"approval_policy":"all_external_contributors"}
```

The public-repo default is `first_time_contributors`, which is **not** sufficient — it lets anyone
who has ever had a PR merged run code on the box unreviewed.

Controls 1 and 2 are the ones actually preventing fork code from executing. Control 3 limits the
blast radius if they ever fail.

### Why not an ephemeral runner

An ephemeral runner de-registers after each job and needs a *fresh* registration token to come
back, which means a long-lived PAT or GitHub App key sitting on a machine other people can reach.
That credential is a larger risk than the one ephemerality removes — ephemeral only isolates jobs
from each other, and every job that reaches this runner has already passed controls 1 and 2. We run
persistent instead, and `actions/checkout` cleans the workspace on every run.

## Setup

Steps marked **sudo** need a password and must be run by hand on the host.

### 1. Create the runner account — *sudo*

```bash
sudo adduser --disabled-password --gecos "GitHub Actions runner" ghrunner
# Deliberately NOT added to sudo, docker, or any group granting privilege.
# CUDA needs no group membership: /dev/nvidia* is world-accessible by default.
sudo -u ghrunner nvidia-smi -L    # confirm the account can see the GPUs
```

### 2. Install the runner — *as ghrunner*

```bash
sudo -u ghrunner -i
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -fsSL -o runner.tar.gz \
  https://github.com/actions/runner/releases/download/v2.336.0/actions-runner-linux-x64-2.336.0.tar.gz
tar xzf runner.tar.gz && rm runner.tar.gz
```

### 3. Register

Generate a registration token (valid one hour) from a machine with `gh` authenticated:

```bash
gh api -X POST repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners/registration-token --jq .token
```

Then, still as `ghrunner`:

```bash
./config.sh --url https://github.com/kylebeggs/MultiDeviceLinearAlgebra.jl \
            --token <TOKEN> \
            --name sasquatch \
            --labels self-hosted,linux,x64,cuda,gpu,sasquatch \
            --work _work \
            --unattended --replace
```

The `cuda` label is what `runs-on: [self-hosted, cuda]` matches. Keep it.

### 4. Install the service — *sudo*

```bash
cd /home/ghrunner/actions-runner
sudo ./svc.sh install ghrunner
sudo ./svc.sh start
sudo ./svc.sh status
```

### 5. Verify

```bash
gh api repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

Should report `"status": "online"` with the `cuda` label present.

## Being a good tenant

The workflow never assumes it owns the machine:

- Every job calls `benchmark/gpu_preflight.sh` before claiming any device. It samples utilisation,
  memory, and resident compute processes over a window — a single reading is not enough, since a
  busy job shows 0% between kernel launches — and picks the best-connected free subset from the
  live `nvidia-smi topo -m` matrix.
- Tests ask for 2 devices, benchmarks for 4. Neither takes the whole complement.
- Benchmarks additionally require host load below 60% of core count. With P2P degraded the halo
  exchange stages through host memory, so a loaded host corrupts exactly the numbers being measured.
- Everything runs `nice -n 19` with `JULIA_NUM_THREADS=4`.
- `benchmark/gpu_watchdog.sh` records load and any foreign process for the run's duration, so a
  contended result is flagged rather than believed.
- `concurrency: gpu-sasquatch` with `cancel-in-progress: false` means GPU jobs never overlap.

**When the host is busy, jobs stand down and say so loudly** — the PR comment states explicitly
that no measurement was taken. A silently missing table is how a device-blind result got mistaken
for a real one once already; a skip must never read as a pass.

To force a run when the load gate is in the way: *Actions → GPU tests and benchmarks → Run
workflow → force*.

## Removing the runner

```bash
sudo /home/ghrunner/actions-runner/svc.sh stop
sudo /home/ghrunner/actions-runner/svc.sh uninstall
sudo -u ghrunner -i
cd ~/actions-runner
./config.sh remove --token "$(gh api -X POST repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners/remove-token --jq .token)"
```

Then `sudo deluser --remove-home ghrunner`. The workflow is harmless with no runner attached: jobs
queue and time out rather than running anywhere unexpected.
