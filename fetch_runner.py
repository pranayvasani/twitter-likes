#!/usr/bin/env python3
import time
import json
from pathlib import Path
import subprocess
import sys

DATA = Path("/app/data")
TOKENS = DATA / "tokens.json"
POLL_INTERVAL = 3   # seconds to wait while tokens not present
POST_FETCH_SLEEP = 2

def wait_for_tokens():
    print("fetch_runner: waiting for data/tokens.json ...")
    while not TOKENS.exists():
        time.sleep(POLL_INTERVAL)
    print("fetch_runner: tokens.json found.")

def run_fetch_script():
    # run the fetch_all_likes_requests.py in the container/process
    print("fetch_runner: starting fetch_all_likes_requests.py ...")
    # Use same Python executable
    cmd = [sys.executable, "fetch_all_likes_requests.py"]
    p = subprocess.Popen(cmd)
    p.wait()
    rc = p.returncode
    print(f"fetch_runner: fetch script exit code: {rc}")
    return rc

def main():
    wait_for_tokens()
    rc = run_fetch_script()
    if rc == 0:
        print("fetch_runner: fetch completed successfully.")
    else:
        print("fetch_runner: fetch finished with non-zero exit code.")
    # optional: pause a little then exit; docker-compose service will finish
    time.sleep(POST_FETCH_SLEEP)
    print("fetch_runner: exiting.")

if __name__ == "__main__":
    main()
