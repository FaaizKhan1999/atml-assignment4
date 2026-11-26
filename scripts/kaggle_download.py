#!/usr/bin/env python3
"""Resilient Kaggle kernel-output downloader for flaky networks.

Uses the installed kaggle client's own listing request, but retries EVERY request separately (each listing page, each
file) instead of aborting the whole download on the first connection error. Files already downloaded completely (atomic
temp + rename, recorded in <dir>/.download_state.json) are skipped. --exclude REGEX skips matching file names (e.g. large
checkpoints that stay on Kaggle). Run with the kaggle CLI's Python: ~/.local/share/uv/tools/kaggle/bin/python.

  python kaggle_download.py OWNER/SLUG OUT_DIR [--exclude '^merged/']
"""
import argparse
import json
import os
import re
import sys
import time

import requests


def retry(fn, what, attempts=40, wait=10):
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # network errors of any kind
            print(f"  {what}: attempt {i + 1}/{attempts} failed ({type(e).__name__}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise SystemExit(f"giving up on {what}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("out")
    ap.add_argument("--exclude", default=None)
    ap.add_argument("--include", default=None, help="only file names matching this regex")
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    api = retry(lambda: (lambda x: (x.authenticate(), x)[1])(KaggleApi()), "authenticate")
    owner, slug = a.kernel.split("/")
    os.makedirs(a.out, exist_ok=True)
    state_p = os.path.join(a.out, ".download_state.json")
    done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
    excl = re.compile(a.exclude) if a.exclude else None
    incl = re.compile(a.include) if a.include else None
    items, token, page = [], None, 0
    log = None
    while True:
        def lst(token=token):
            with api.build_kaggle_client() as k:
                r = ApiListKernelSessionOutputRequest()
                r.user_name, r.kernel_slug, r.page_size = owner, slug, 100
                if token:
                    r.page_token = token
                return k.kernels.kernels_api_client.list_kernel_session_output(r)
        resp = retry(lst, f"list page {page}")
        items += list(resp.files or [])
        log = log or resp.log
        token = resp.next_page_token
        page += 1
        if not token:
            break
    print(f"{len(items)} output files listed; {len(done)} already done", flush=True)
    root = os.path.realpath(a.out)
    n_new = n_skip = 0
    for it in items:
        name = it.file_name
        if (excl and excl.search(name)) or (incl and not incl.search(name)):
            n_skip += 1
            continue
        if name in done:
            continue
        dst = os.path.join(a.out, name)
        assert os.path.realpath(dst).startswith(root + os.sep), name
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        def get(url=it.url, dst=dst):
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            tmp = dst + ".part"
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    n += len(chunk)
            exp = r.headers.get("Content-Length")
            if exp is not None and int(exp) != n:
                raise IOError(f"short read {n}/{exp}")
            os.replace(tmp, dst)
        retry(get, name)
        done.add(name)
        json.dump(sorted(done), open(state_p, "w"))
        n_new += 1
        print(f"  ok {name}", flush=True)
    if log:
        with open(os.path.join(a.out, slug + ".log"), "w", encoding="utf-8") as f:
            f.write(log)
    print(f"done: {n_new} new, {len(done)} total, {n_skip} excluded", flush=True)


if __name__ == "__main__":
    sys.exit(main())
