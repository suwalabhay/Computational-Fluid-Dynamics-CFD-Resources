#!/usr/bin/env python3
"""Run every ``scripts/**/main.py`` headless and report the ones that fail.

Each script is run from its own directory as::

    python main.py --no-show --output <tmpdir> [--steps N]

``--steps`` is passed only to scripts whose ``--help`` output lists it. A
script passes when it exits with status 0 within the timeout and writes at
least one PNG into ``<tmpdir>``.

Usage::

    python tools/run_scripts.py                 # all scripts
    python tools/run_scripts.py pod lid_driven  # scripts whose path contains any filter
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def discover(filters):
    scripts = sorted((ROOT / "scripts").glob("*/*/main.py"))
    if filters:
        scripts = [s for s in scripts if any(f in str(s.parent) for f in filters)]
    return scripts


def run(cmd, cwd, env, timeout):
    """Run cmd in its own process group so children die with it on timeout."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        return None, out, True


def run_script(script, steps, timeout, out_root):
    rel = script.parent.relative_to(ROOT)
    env = dict(
        os.environ,
        MPLBACKEND="Agg",
        SDL_VIDEODRIVER="dummy",
        SDL_AUDIODRIVER="dummy",
        PYGAME_HIDE_SUPPORT_PROMPT="1",
    )
    start = time.monotonic()

    code, help_text, timed_out = run(
        [sys.executable, "main.py", "--help"], script.parent, env, timeout
    )
    if timed_out or code != 0:
        return rel, False, time.monotonic() - start, "`--help` failed:\n" + help_text

    out_dir = Path(out_root) / str(rel).replace(os.sep, "__")
    cmd = [sys.executable, "main.py", "--no-show", "--output", str(out_dir)]
    if "--steps" in help_text:
        cmd += ["--steps", str(steps)]

    code, output, timed_out = run(cmd, script.parent, env, timeout)
    elapsed = time.monotonic() - start
    if timed_out:
        return rel, False, elapsed, f"timed out after {timeout}s\n{output}"
    if code != 0:
        return rel, False, elapsed, f"exit code {code}\n{output}"
    if not list(out_dir.glob("*.png")):
        return rel, False, elapsed, f"no PNG written to --output\n{output}"
    return rel, True, elapsed, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("filters", nargs="*", help="substrings of script paths to run")
    parser.add_argument("--steps", type=int, default=10, help="value passed to --steps")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per script")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = parser.parse_args(argv)

    scripts = discover(args.filters)
    if not scripts:
        print("no scripts matched")
        return 1

    failures = []
    with (
        tempfile.TemporaryDirectory() as out_root,
        ThreadPoolExecutor(args.jobs) as pool,
    ):
        jobs = [
            pool.submit(run_script, s, args.steps, args.timeout, out_root)
            for s in scripts
        ]
        for job in jobs:
            rel, ok, elapsed, output = job.result()
            print(f"{'PASS' if ok else 'FAIL'}  {elapsed:6.1f}s  {rel}", flush=True)
            if not ok:
                failures.append((rel, output))

    for rel, output in failures:
        tail = "\n".join(output.strip().splitlines()[-20:])
        print(f"\n--- {rel} ---\n{tail}")
    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} scripts passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
