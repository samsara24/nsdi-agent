"""诊断多卡 NCCL 通信是否真的可用。

`nvidia-smi topo -p2p r` 报告 OK 只说明驱动认为可以，不代表数据真能过去：
主板 ACS 打开时，P2P 写会被静默重定向，NCCL 集合通信会挂起而不是报错。
vLLM 的张量并行卡在初始化或第一次 all-reduce 时，先跑这个脚本定位，
不要在 32B 模型上反复试——加载一次就要三分钟。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world_size: int, size_mb: int, result) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29591")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    tensor = torch.ones(size_mb * 256 * 1024, dtype=torch.float32, device=f"cuda:{rank}")
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    expected = float(world_size)
    ok = bool(tensor[0].item() == expected and tensor[-1].item() == expected)
    result[rank] = 1 if ok else 0
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--size-mb", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(全部)")
    print(f"CUDA_VISIBLE_DEVICES={visible}  world_size={args.world_size}  张量={args.size_mb}MB")
    print(f"NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', '(未设置)')}")

    result = mp.Manager().dict()
    context = mp.spawn(worker, args=(args.world_size, args.size_mb, result),
                       nprocs=args.world_size, join=False)

    # `ProcessContext.join(timeout)` 只要有一个子进程退出就返回，不代表全部完成，
    # 因此必须自己盯到超时为止，否则会把「一个进程崩了」误报成挂起。
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if all(not process.is_alive() for process in context.processes):
            break
        time.sleep(1.0)
    alive = [p.pid for p in context.processes if p.is_alive()]
    exit_codes = [p.exitcode for p in context.processes]
    if alive:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        print(f"结果: 挂起（{args.timeout}s 内没完成 all-reduce），仍存活 pid={alive}"
              f" —— 这条通信路径不可用")
        sys.exit(2)
    if any(code != 0 for code in exit_codes):
        print(f"结果: 子进程异常退出 exitcode={exit_codes} —— 见上方报错")
        sys.exit(4)
    values = [result.get(rank) for rank in range(args.world_size)]
    if all(value == 1 for value in values):
        print("结果: 通过，all-reduce 数值正确")
        sys.exit(0)
    print(f"结果: 数值错误 {values} —— P2P 传输被静默破坏，比挂起更危险")
    sys.exit(3)


if __name__ == "__main__":
    main()
