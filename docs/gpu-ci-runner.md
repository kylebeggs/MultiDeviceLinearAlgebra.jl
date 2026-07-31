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

`scripts/setup_gpu_runner.sh` does all of it. It needs sudo for two things — creating the account
and installing the systemd service — so it must be run **by hand, on the host**, by someone who can
authenticate.

First, on a machine where `gh` is authenticated, mint a registration token. It is valid for one
hour and is not a long-lived credential:

```bash
gh api -X POST repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners/registration-token --jq .token
```

Then, on the host, from a checkout of this repository:

```bash
./scripts/setup_gpu_runner.sh --token <TOKEN>
```

It creates the unprivileged `ghrunner` account (no sudo, no docker, no privileged groups — CUDA
needs none, `/dev/nvidia*` is world-accessible), confirms that account can actually see the GPUs,
resolves the current `actions/runner` release, **verifies the published SHA256 before extracting
anything**, registers with the labels below, and installs and starts the service.

```
self-hosted,linux,x64,cuda,gpu,sasquatch
```

The `cuda` label is what `runs-on: [self-hosted, cuda]` matches. Keep it.

The runner version is resolved at install time rather than pinned here: the release moves, a
persistent runner self-updates regardless, and a stale version pinned in a document is how you end
up installing something that no longer registers. Pass `--version` to override.

### Verify

```bash
gh api repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners --jq '.runners[] | {name, status, labels: [.labels[].name]}'
```

Should report `"status": "online"` with the `cuda` label present.

Then check the preflight guard against this host's real topology, and run its test suite — the
selection and wait logic needs bash 4+, so it cannot be exercised on a Mac dev box:

```bash
./benchmark/gpu_preflight.sh --need 2 --prefer-far --bare   # should cross a NUMA node
./benchmark/gpu_preflight.sh --need 4 --bare                # should stay on one socket
./benchmark/test_gpu_preflight.sh                           # mocked nvidia-smi, no real devices
```

## Being a good tenant

The workflow never assumes it owns the machine, and **never takes a GPU someone else is on.**

- Every job calls `benchmark/gpu_preflight.sh` before claiming any device. It samples utilisation,
  memory, and resident compute processes over a window — a single reading is not enough, since a
  busy job shows 0% between kernel launches — and a resident compute process disqualifies a device
  outright even at 0%, because someone between phases of their job still owns it.
- **Busy means wait, not barge in.** A job re-probes every 2 minutes for up to 2 hours
  (`GPU_WAIT_SECONDS`), then stands down. Waiting holds no device — it only re-reads `nvidia-smi`,
  so a job that waits the full window costs another user nothing. The alternative is worse in both
  directions: giving up immediately means a busy week produces no GPU coverage at all, and taking a
  device that is in use is not on the table.
- Tests ask for 2 devices, benchmarks for 4. Neither takes the whole complement.
- `JULIA_CUDA_HARD_MEMORY_LIMIT=16GiB` caps the pool. MDLA sizes itself against whatever is free,
  so an uncapped run is the one way this workflow could squeeze a co-tenant that starts *after* it
  did.
- Benchmarks additionally require host load below 60% of core count. With P2P degraded the halo
  exchange stages through host memory, so a loaded host corrupts exactly the numbers being measured.
- Everything runs `nice -n 19` with `JULIA_NUM_THREADS=4`.
- `benchmark/gpu_watchdog.sh` records load and any foreign process for the run's duration, so a
  contended result is flagged rather than believed. It deliberately does **not** abort mid-run: our
  jobs are minutes long, the memory cap already bounds what we can take, and killing a run
  mid-kernel mostly wastes the GPU time already spent.
- `concurrency: gpu-sasquatch` with `cancel-in-progress: false` means GPU jobs never overlap.
- A job whose pull request gets a newer commit while it is still *waiting* abandons the wait
  (preflight exit 4) and hands the host to the newer run, instead of eventually testing a stale
  commit. `cancel-in-progress: false` would otherwise leave both sitting on the queue.

**When the wait runs out, jobs stand down and say so loudly** — the PR comment states explicitly
that no measurement was taken. A silently missing table is how a device-blind result got mistaken
for a real one once already; a skip must never read as a pass.

Which devices a job takes is not arbitrary either. Benchmarks take the **best-connected** free
subset, since an accidental cross-socket set makes the halo exchange look far worse than the
hardware can do. Tests pass `--prefer-far` and take the **worst-connected** one, because
`test/test_cross_socket.jl` exists to exercise the `SYS`-class links that a mistranslating IOMMU
corrupts hardest — handed a well-connected pair, `_far_device_pair()` reports `cross_numa = false`
and that whole file quietly degrades into re-testing an adjacent pair.

To force a run when the load gate is in the way: *Actions → GPU tests and benchmarks → Run
workflow → force*.

## Removing the runner

```bash
./scripts/setup_gpu_runner.sh --uninstall --token "$(gh api -X POST \
  repos/kylebeggs/MultiDeviceLinearAlgebra.jl/actions/runners/remove-token --jq .token)"
```

Note that this needs a **remove** token, not a registration token. The account is left in place;
drop it with `sudo deluser --remove-home ghrunner`. The workflow is harmless with no runner
attached: jobs queue and time out rather than running anywhere unexpected.

## If the repository moves

Issue #11 also floats transferring this repository to the JuliaGPU organisation. The runner is
registered against a repository URL, so a transfer means re-registering — uninstall as above, then
re-run the setup script with `--repo <new/owner>`. Nothing else in the workflow is
repository-specific.
