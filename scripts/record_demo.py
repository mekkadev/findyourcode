#!/usr/bin/env python3
"""Record the README demo.

    pip install asciinema && cargo install --git https://github.com/asciinema/agg
    python scripts/record_demo.py

Types each command at human speed and runs it for real inside a pty, so the
output in the recording is the tool's output, not a mock-up. Writes
docs/demo.cast and docs/demo.gif.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "examples" / "demo_repo"
OUT = ROOT / "docs"

SCRIPT = [
    ("fyc index", 1.0),
    ('fyc find "reject a request without a valid ticket" -n 2 -L 4', 3.0),
    ('fyc find "how do we take money from a card" -n 3 -f paths', 2.2),
    ("fyc doctor", 2.4),
]

PROMPT = "\033[38;5;114m\u276f\033[0m "


def play() -> None:
    """Runs inside the recording: this is what the viewer sees."""
    os.chdir(DEMO)
    shutil.rmtree(DEMO / ".findyourcode", ignore_errors=True)
    time.sleep(0.6)

    for command, pause in SCRIPT:
        sys.stdout.write(PROMPT)
        sys.stdout.flush()
        for char in command:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(0.035)
        time.sleep(0.35)
        sys.stdout.write("\n")
        sys.stdout.flush()

        subprocess.run(command, shell=True, check=False)
        time.sleep(pause)

    sys.stdout.write(PROMPT + "\n")
    sys.stdout.flush()
    time.sleep(1.2)


def record(args) -> int:
    for tool in ("asciinema", "agg"):
        if shutil.which(tool) is None:
            print(f"{tool} not found on PATH", file=sys.stderr)
            return 2

    OUT.mkdir(exist_ok=True)
    cast, gif = OUT / "demo.cast", OUT / "demo.gif"
    cast.unlink(missing_ok=True)

    env = {**os.environ, "FYC_DEMO_CHILD": "1", "COLUMNS": str(args.cols), "LINES": str(args.rows)}
    subprocess.run(
        [
            "asciinema",
            "rec",
            str(cast),
            "--overwrite",
            "--quiet",
            "--cols",
            str(args.cols),
            "--rows",
            str(args.rows),
            "--command",
            f"{sys.executable} {__file__}",
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            "agg",
            str(cast),
            str(gif),
            "--font-size",
            "16",
            "--speed",
            "1.0",
            "--fps-cap",
            "20",
            "--theme",
            args.theme,
            "--idle-time-limit",
            "2",
        ],
        check=True,
    )
    print(f"{gif} ({gif.stat().st_size / 1e6:.1f} mb)")
    return 0


def main() -> int:
    if os.environ.get("FYC_DEMO_CHILD"):
        play()
        return 0

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--rows", type=int, default=28)
    parser.add_argument(
        "--theme",
        default=("1e1e2e,cdd6f4,45475a,f38ba8,a6e3a1,f9e2af,89b4fa,f5c2e7,94e2d5,bac2de"),
    )
    return record(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
