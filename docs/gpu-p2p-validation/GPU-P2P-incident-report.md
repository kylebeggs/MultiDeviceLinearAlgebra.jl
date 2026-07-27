# Incident + Diagnosis: GPU peer-to-peer is broken on `sasquatch`

**Host:** `sasquatch` — Ubuntu 24.04.4 LTS, kernel 6.17.0-35-generic
**Hardware:** 9× NVIDIA A30 (24 GB), 2× Intel Xeon Platinum 8592V, 4 NUMA nodes
**Driver:** 535.309.01 (CUDA 12.2)
**Booted:** 2026-07-13 23:57 (uptime 13 days at time of writing)
**Reported by:** kbeggs · **Date:** 2026-07-27

---

## TL;DR

1. **GPU-to-GPU (peer-to-peer) transfers on this host silently return corrupted data.** No error is
   raised. Any multi-GPU job on this machine has been producing wrong numbers, quietly.
2. **Root cause: the IOMMU is enabled in translating mode.** All 9 GPUs report
   `iommu_group/type = DMA-FQ`, and the kernel cmdline has neither `iommu=pt` nor `intel_iommu=off`.
   NVIDIA requires VT-d disabled or in passthrough for PCIe P2P.
3. **The NVIDIA driver is currently wedged** and needs a reboot. Multiple users' processes are stuck
   in uninterruptible sleep and cannot be killed.
4. **Fix and recovery are the same action:** add `intel_iommu=off` to the kernel cmdline and reboot.

---

## Part 1 — Immediate: the driver is wedged

`nvidia-smi` is unresponsive. Processes are accumulating in uninterruptible (`D`) state inside the
NVIDIA driver. They cannot be killed, and the `nvidia` modules cannot be unloaded while they hold
references, so **a reboot is required**.

Currently stuck:

| Owner | Process | Stuck since |
|---|---|---|
| `netdata` | `nvidia-smi` | ~10:33 — **predates any diagnostic work** |
| `kd7yjw` | 3× `julia` (CUDA workload) | ~12:08 |
| `sasquatch` | `nvidia-smi` (monitoring agent) | accumulating continuously, now several per minute |

**Attribution, stated plainly:** the `netdata` process has been stuck since roughly 10:33, which is
before any diagnostic work began on this host (first command 11:22, the P2P test sweep 12:05). The
driver was therefore already in a degraded state. That said, the P2P test sweep at 12:05 is what
took it from degraded to fully unresponsive, and `kd7yjw`'s jobs froze at 12:08 as a direct
consequence. Please give `kd7yjw` warning before rebooting — their work is already lost, but they
should know why.

During one brief window the driver responded and reported `[Unknown Error]` for GPU 3's utilization
— GPU 3 being one of the two GPUs in the test pair that hung. It has since stopped responding again.

Note also that the monitoring agent is retry-looping and adding to the pile. Worth pausing whatever
polls `nvidia-smi` before the reboot so it doesn't immediately re-wedge on a machine that still has
the underlying fault.

---

## Part 2 — Diagnosis: peer-to-peer is broken on every GPU pair

Tested with NVIDIA's own `cuda-samples` v12.0 `simpleP2P`, which performs a GPU→GPU transfer and
verifies every element against a reference.

| Pair | Topology | Peer bandwidth | Data verification | Result |
|---|---|---|---|---|
| 0 ↔ 1 | `PIX` — same PCIe switch, same NUMA node | **1.07 GB/s** | every element `nan` | **FAIL** |
| 0 ↔ 2 | `NODE` — same socket, across host bridges | **0.97 GB/s** | every element `nan` | **FAIL** |
| 0 ↔ 3 | `NODE` | — | — | **hung; wedged the driver** |

Two independent things are wrong:

**The transfers are not actually peer-to-peer.** Expected same-switch P2P bandwidth on PCIe Gen4 x16
is 20–25 GB/s. The measured ~1 GB/s is the signature of every transfer being routed the long way
through host memory under IOMMU translation.

**The data arrives corrupted, with no error.** Every verified element came back `nan`. A separate
run of the same test also produced a hard fault when a kernel dereferenced a peer pointer:

```
CUDA error at simpleP2P.cu:200 code=719(cudaErrorLaunchFailure) "cudaDeviceSynchronize()"
```

So the memcpy path silently yields garbage and the direct kernel-load path faults outright.

**Critically, pair 0↔1 is the most favourable pair on the machine** — same PCIe switch, same NUMA
node, no cross-socket hop. This rules out the usual suspects: it is not a cross-socket/UPI
limitation, not a single bad GPU, and not a bad switch port. The fault is host-wide.

Testing was halted after 3 of 36 pairs: the verdict was already unambiguous and the driver had hung.

### Aggravating factor: the driver advertises P2P that does not work

`cudaDeviceCanAccessPeer` returns **true for all 72 ordered pairs**, and `nvidia-smi topo -p2p r`
and `-p2p w` report `OK` for every pair. Applications trust these flags, take the P2P fast path,
and get corrupted data. This is why the failure is silent rather than loud.

(`nvidia-smi topo -p2p a` does report atomics as `NS` — not supported — on every pair.)

---

## Part 3 — Root cause

The IOMMU is active and translating DMA for the GPUs:

```
$ cat /sys/bus/pci/devices/0000:2b:00.0/iommu_group/type
DMA-FQ                    # identical on all 9 GPUs

$ cat /proc/cmdline
BOOT_IMAGE=/boot/vmlinuz-6.17.0-35-generic root=UUID=... ro quiet splash vt.handoff=7
                          # no iommu=pt, no intel_iommu=off
```

NVIDIA requires VT-d to be **disabled or in passthrough mode** for PCIe peer-to-peer to function.
Under active translation (`DMA-FQ`), peer writes carry addresses the target GPU never mapped. The
observed combination — P2P advertised as available, ~1 GB/s throughput, `nan` payloads, and a
launch fault on direct peer access — is the textbook presentation.

**The hardware itself is healthy.** This is a configuration fault, not a failing card:

- 0 uncorrected ECC errors, 0 corrected (aggregate), across all 9 GPUs
- 0 remapped rows, no remapping failures pending
- No `Xid` or `NVRM` errors in the kernel journal
- All 9 links negotiated at full width and speed: PCIe Gen4 x16
- Topology is as expected: no NVLink; GPUs 0–3 on NUMA 0, GPUs 4–8 on NUMA 2, behind Broadcom/PLX
  switches

---

## Part 4 — Fix

```bash
# /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=off"

sudo update-grub
sudo reboot
```

`iommu=pt` (passthrough) is the alternative and is preferable if anything on this host needs the
IOMMU for device isolation or VFIO passthrough — it keeps the IOMMU present but stops it translating
for these devices. If nothing on the box needs VFIO, `intel_iommu=off` is the simpler and more
certain option.

Check the BIOS as well: if VT-d is enabled there it can be disabled at that level instead.

### Also worth checking while you're in there: PCIe ACS

ACS (Access Control Services) on the upstream Broadcom/PLX switch ports forces peer traffic up to
the root complex and is the classic partner to this failure mode. It needs root to read, so it was
not checked:

```bash
sudo lspci -vvv | grep -i -B12 'ACSCtl' | grep -E 'ACSCtl|^[0-9a-f]{2}:'
```

If `ACSCtl` shows `SrcValid+`, `RR+`, or `EC+` on the bridges upstream of the GPUs, disable ACS in
BIOS (often "ACS Enable" / "PCIe ACS") for full P2P bandwidth.

---

## Part 5 — Verification after reboot

The NVIDIA test binaries are already built at `~kbeggs/dev/gpu-comms-validation/`. No build step
needed.

**1. Confirm the IOMMU setting took:**

```bash
cat /proc/cmdline | grep -o 'intel_iommu=[a-z]*'
cat /sys/bus/pci/devices/0000:2b:00.0/iommu_group/type    # should no longer be DMA-FQ
```

**2. Re-run the pair that failed (takes seconds):**

```bash
cd ~kbeggs/dev/gpu-comms-validation
CUDA_VISIBLE_DEVICES=0,1 ./cuda-samples/Samples/0_Introduction/simpleP2P/simpleP2P
```

Pass looks like: bandwidth **20–25 GB/s**, and `Test passed!` with no verification errors. Anything
near 1 GB/s, or any `Verification error`, means it is not fixed.

**3. Full 36-pair sweep** (only after step 2 passes — do not run this on a machine that still has
the fault, as that is what hung the driver):

```bash
for i in $(seq 0 8); do for j in $(seq $((i+1)) 8); do
  echo "=== pair $i,$j ==="
  CUDA_VISIBLE_DEVICES=$i,$j ./cuda-samples/Samples/0_Introduction/simpleP2P/simpleP2P 2>&1 \
    | grep -E 'GB/s|Verification error|Test (passed|failed)'
done; done
```

**4. Collective correctness across all 9 GPUs** (NCCL, with per-element checking):

```bash
export LD_LIBRARY_PATH=~kbeggs/dev/gpu-comms-validation/nccl/lib:$LD_LIBRARY_PATH
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 4 -g 9 -c 1 -n 20
```

The `#wrong` column must be `0` on every row.

---

## Appendix — supporting data

Full artifacts in `~kbeggs/dev/gpu-comms-validation/results/`:

| File | Contents |
|---|---|
| `00-baseline.txt` | complete `nvidia-smi -q`, topology matrix, all four `-p2p` capability matrices, per-GPU IOMMU group types, PCIe tree |
| `02-simpleP2P-all-pairs.txt` | raw output of the three completed pair tests |
| `SUMMARY.md` | technical summary |

### Topology reference

```
        GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7 GPU8   NUMA
GPU0     X   PIX  NODE NODE  SYS  SYS  SYS  SYS  SYS     0
GPU1    PIX   X   NODE NODE  SYS  SYS  SYS  SYS  SYS     0
GPU2    NODE NODE  X   PIX   SYS  SYS  SYS  SYS  SYS     0
GPU3    NODE NODE PIX   X    SYS  SYS  SYS  SYS  SYS     0
GPU4     SYS  SYS  SYS  SYS   X   PIX  NODE NODE NODE    2
GPU5     SYS  SYS  SYS  SYS  PIX   X   NODE NODE NODE    2
GPU6     SYS  SYS  SYS  SYS  NODE NODE  X   PIX  PIX     2
GPU7     SYS  SYS  SYS  SYS  NODE NODE PIX   X   PIX     2
GPU8     SYS  SYS  SYS  SYS  NODE NODE PIX  PIX   X      2
```

No NVLink present — all GPU-to-GPU traffic is PCIe.

### One inventory discrepancy to confirm

Internal benchmark notes on this host describe it as a "Multi-GPU server (10x NVIDIA GPUs)", while
`nvidia-smi` enumerates exactly 9 and every recorded result references 9. If the chassis is
physically populated with 10 A30s, one has dropped off the PCIe bus and should be investigated
separately. If it was always 9, the note is simply wrong and can be ignored.
