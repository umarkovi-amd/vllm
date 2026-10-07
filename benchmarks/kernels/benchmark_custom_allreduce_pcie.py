# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate and benchmark vLLM custom all-reduce in the experimental ROCm PCIe mode.

For each message size (tokens x hidden, bf16) this reports, on all visible GPUs:
  * whether CustomAllreduce takes the message (below VLLM_ROCM_CUSTOM_AR_PCIE_MAX_SIZE)
  * correctness under HIP/CUDA graph capture with the producer *inside* the graph
    (``t.copy_(src)`` immediately before each all-reduce, no sync), which is the decode
    pattern that exposes stale peer reads; every output is compared with the exact
    fp32 sum of all ranks' inputs (custom all-reduce accumulates in fp32 and must
    match it bit-exactly after rounding)
  * graph-replay latency per call: custom all-reduce vs RCCL/NCCL (torch.distributed)

Usage (one process per visible GPU is spawned; run from outside the vLLM source tree,
otherwise spawned workers import the uncompiled source package):
  cd /tmp && VLLM_ROCM_CUSTOM_AR_PCIE=1 python \\
      <vllm>/benchmarks/kernels/benchmark_custom_allreduce_pcie.py
  # reproduce the zero-copy hazard (expect mismatches on non-coherent PCIe):
  cd /tmp && VLLM_ROCM_CUSTOM_AR_PCIE=1 python \\
      <vllm>/benchmarks/kernels/benchmark_custom_allreduce_pcie.py --mode zero-copy
Options: --tokens 1,2,4,8,16  --hidden 6656  --replays 20  --bufs 50  --calls 100
"""

import argparse
import contextlib
import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _graph_time_us(fn, calls: int, capture_ctx=None, trials: int = 7) -> float:
    """Capture `fn` in a graph (inside `capture_ctx`, e.g. CustomAllreduce.capture())
    and time replays *after* leaving the context, as vLLM does."""
    ctx = capture_ctx if capture_ctx is not None else contextlib.nullcontext()
    with ctx:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / calls)
    del graph
    return statistics.median(samples)


def _worker(rank: int, world: int, args: argparse.Namespace) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, device_id=torch.device(f"cuda:{rank}")
    )
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    ca = CustomAllreduce(group=cpu_group, device=torch.device(f"cuda:{rank}"))
    if ca.disabled:
        raise RuntimeError(
            "CustomAllreduce is disabled; set VLLM_ROCM_CUSTOM_AR_PCIE=1 on PCIe "
            "systems (or run on a platform where it is natively supported)."
        )
    if args.mode == "zero-copy":
        if not hasattr(ca, "_capture_registered"):
            raise RuntimeError(
                "--mode zero-copy needs a vLLM with register_graph_buffers support"
            )
        ca._capture_registered = True  # reproduce the pre-fix behaviour
    cap = getattr(ca, "_pcie_max_size", getattr(ca, "_pcie_ar_max", None))
    gen = torch.Generator(device="cuda").manual_seed(1234 + rank)

    def log(msg: str) -> None:
        if args.verbose and rank == 0:
            print(f"[rank0] {msg}", flush=True)

    rows = []
    for tokens in args.tokens:
        log(f"tokens={tokens}: correctness")
        bufs = [
            torch.empty(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
            for _ in range(args.bufs)
        ]
        srcs = [torch.empty_like(b) for b in bufs]
        if not ca.should_custom_ar(bufs[0]):
            rows.append((tokens, bufs[0].nbytes, "rccl", None, None, None))
            continue

        # Correctness: producer inside the graph, fresh data every replay.
        with ca.capture():
            for b, s in zip(bufs[:3], srcs[:3]):
                s.normal_(generator=gen)
                b.copy_(s)
                ca.custom_all_reduce(b)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = []
                for b, s in zip(bufs, srcs):
                    b.copy_(s)
                    outs.append(ca.custom_all_reduce(b))
        mismatches = 0
        for _ in range(args.replays):
            for s in srcs:
                s.normal_(generator=gen)
            torch.cuda.synchronize()
            dist.barrier()
            graph.replay()
            torch.cuda.synchronize()
            for s, o in zip(srcs, outs):
                gathered = [torch.empty_like(s) for _ in range(world)]
                dist.all_gather(gathered, s)
                ref = torch.stack(gathered).float().sum(0).to(torch.bfloat16)
                mismatches += int(not torch.equal(o, ref))
        del graph

        log(f"tokens={tokens}: mismatches={mismatches}; latency")
        # Latency: back-to-back calls in one graph, custom vs RCCL.
        x = torch.zeros(tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
        custom_us = _graph_time_us(
            lambda x=x: [ca.custom_all_reduce(x) for _ in range(args.calls)],
            args.calls,
            capture_ctx=ca.capture(),
        )
        rccl_us = _graph_time_us(
            lambda x=x: [dist.all_reduce(x) for _ in range(args.calls)], args.calls
        )
        stats = torch.tensor([mismatches, custom_us, rccl_us], device="cuda")
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        rows.append(
            (
                tokens,
                bufs[0].nbytes,
                "custom",
                int(stats[0]),
                stats[1].item(),
                stats[2].item(),
            )
        )

    if rank == 0:
        print(
            f"\nworld={world} device={torch.cuda.get_device_name()} "
            f"arch={torch.cuda.get_device_properties(0).gcnArchName} mode={args.mode} "
            f"cap={cap} calls/size={args.replays * args.bufs}"
        )
        print(
            f"{'tokens':>6} {'bytes':>9} {'path':>6} {'mismatch':>9} "
            f"{'custom_us':>10} {'rccl_us':>9}"
        )
        for t, nbytes, path, bad, cus, rus in rows:
            if path == "rccl":
                print(f"{t:>6} {nbytes:>9} {path:>6} {'-':>9} {'-':>10} {'-':>9}")
            else:
                print(f"{t:>6} {nbytes:>9} {path:>6} {bad:>9} {cus:>10.2f} {rus:>9.2f}")
    ca.close()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--tokens",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--hidden", type=int, default=6656)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--bufs", type=int, default=50)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--mode", choices=["copy-in", "zero-copy"], default="copy-in")
    parser.add_argument("--port", type=int, default=29731)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    world = torch.cuda.device_count()
    if world < 2:
        raise SystemExit("needs at least 2 visible GPUs")
    mp.spawn(_worker, args=(world, args), nprocs=world)


if __name__ == "__main__":
    main()
