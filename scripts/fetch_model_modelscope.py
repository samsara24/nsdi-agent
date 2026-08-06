#!/usr/bin/env python
"""Fetch a model snapshot from ModelScope with size verification and retries.

HuggingFace is unreachable from this host, and the local
DeepSeek-R1-Distill-Qwen-32B directory only holds one truncated shard, so the
snapshot has to be completed before any vLLM run.  Files whose on-disk size
already matches the remote manifest are skipped, which makes the script safe to
re-run after an interruption.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

API = "https://modelscope.cn/api/v1/models"
SKIP_SUFFIXES = (".md", ".gitattributes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def list_files(model_id: str, revision: str) -> List[Dict[str, Any]]:
    url = f"{API}/{model_id}/repo/files?Revision={revision}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    files = payload.get("Data", {}).get("Files", [])
    return [
        item for item in files
        if int(item.get("Size", 0)) > 0 and not item["Path"].endswith(SKIP_SUFFIXES)
    ]


def download(model_id: str, revision: str, item: Dict[str, Any], output: Path, retries: int) -> str:
    target = output / item["Path"]
    expected = int(item["Size"])
    if target.exists() and target.stat().st_size == expected:
        return f"skip {item['Path']} (already complete)"
    target.parent.mkdir(parents=True, exist_ok=True)
    quoted = urllib.parse.quote(item["Path"])
    url = f"{API}/{model_id}/repo?Revision={revision}&FilePath={quoted}"
    partial = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            started = time.time()
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
                while True:
                    chunk = response.read(8 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
            size = partial.stat().st_size
            if size != expected:
                raise IOError(f"size mismatch: got {size}, expected {expected}")
            partial.replace(target)
            rate = size / max(1e-6, time.time() - started) / 1e6
            return f"done {item['Path']} ({size / 1e9:.2f}GB, {rate:.1f}MB/s)"
        except Exception as error:  # noqa: BLE001 - retry any transport failure
            partial.unlink(missing_ok=True)
            if attempt == retries:
                return f"FAIL {item['Path']}: {error}"
            time.sleep(min(60, 5 * attempt))
            print(f"[fetch] retry {attempt} for {item['Path']}: {error}", flush=True)
    return f"FAIL {item['Path']}: exhausted retries"


def main() -> int:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = list_files(args.model_id, args.revision)
    total = sum(int(item["Size"]) for item in files)
    print(f"[fetch] {len(files)} files, {total / 1e9:.2f}GB total -> {output}", flush=True)

    failures = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download, args.model_id, args.revision, item, output, args.retries): item
            for item in files
        }
        for future in as_completed(futures):
            message = future.result()
            print(f"[fetch] {message}", flush=True)
            if message.startswith("FAIL"):
                failures.append(message)

    elapsed = time.time() - started
    print(f"[fetch] finished in {elapsed / 60:.1f} min", flush=True)
    if failures:
        print(f"[fetch] {len(failures)} file(s) failed", flush=True)
        return 1

    missing = [
        item["Path"] for item in files
        if not (output / item["Path"]).exists()
        or (output / item["Path"]).stat().st_size != int(item["Size"])
    ]
    if missing:
        print(f"[fetch] incomplete after run: {missing}", flush=True)
        return 1
    print("[fetch] snapshot verified complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
