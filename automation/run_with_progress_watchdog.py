#!/usr/bin/env python3
"""Run a resumable command and abort it only after sustained zero file growth."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except FileNotFoundError:
            # Download workers atomically rename and remove part files.
            continue
    return total


def terminate_group(process: subprocess.Popen[bytes], grace_seconds: float = 20) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-path", type=Path, action="append", required=True)
    parser.add_argument("--stall-seconds", type=float, default=600)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--minimum-growth-bytes", type=int, default=1024 * 1024)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if (
        args.stall_seconds <= 0
        or args.poll_seconds <= 0
        or args.minimum_growth_bytes <= 0
    ):
        parser.error("watchdog durations must be positive")

    for watch_path in args.watch_path:
        watch_path.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, start_new_session=True)
    last_size = sum(tree_size(path) for path in args.watch_path)
    last_growth = time.monotonic()
    pending_growth = 0
    print(
        f"watchdog: pid={process.pid} initial_bytes={last_size} "
        f"stall_seconds={args.stall_seconds:g}",
        flush=True,
    )
    try:
        while process.poll() is None:
            time.sleep(args.poll_seconds)
            current_size = sum(tree_size(path) for path in args.watch_path)
            delta = current_size - last_size
            last_size = current_size
            if delta > 0:
                pending_growth += delta
                print(
                    f"watchdog: bytes={current_size} delta={delta} "
                    f"credited={pending_growth}",
                    flush=True,
                )
            if pending_growth >= args.minimum_growth_bytes:
                last_growth = time.monotonic()
                pending_growth = 0
            stalled_for = time.monotonic() - last_growth
            if stalled_for >= args.stall_seconds:
                print(
                    f"watchdog: no file growth for {stalled_for:.0f}s; "
                    "terminating resumable worker",
                    file=sys.stderr,
                    flush=True,
                )
                terminate_group(process)
                return 75
        return int(process.returncode or 0)
    except (KeyboardInterrupt, SystemExit):
        terminate_group(process)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
