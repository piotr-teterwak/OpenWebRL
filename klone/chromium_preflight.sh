#!/bin/bash
# Chromium launch preflight: warm-up, then N-way with retries.
#
# The naive version (one 8-way launch, 120s, fail hard) rejects healthy nodes.
# The headless shell is rsynced to /scr immediately before, so N simultaneous
# cold launches all fault in the same ~180MB binary at once; on a busy node that
# alone can blow a 120s deadline. A single sequential launch primes the page
# cache first, and only a repeated N-way failure counts as the node being unable
# to run browsers.
#
# MUST run through browser_python.sh -- it clears LD_LIBRARY_PATH exactly as the
# real rollout subprocesses do. Bare python resolves 35 libs off GPFS and gives
# results that do not reflect what training will actually do.
PROJ="${PROJ:-/gscratch/cse/piotrt}"
N="${1:-8}"

run() {  # concurrency, per-launch timeout ms, wall-clock seconds
  timeout "$3" "$PROJ/browser_python.sh" - "$1" "$2" <<'PYEOF'
import asyncio, sys, time
from playwright.async_api import async_playwright
N = int(sys.argv[1]); TMO = int(sys.argv[2])
ARGS = ["--disable-gpu","--no-sandbox","--disable-dev-shm-usage","--disable-logging",
        "--ignore-certificate-errors","--log-level=3","--silent"]
async def one(p):
    b = await p.chromium.launch(headless=True, args=ARGS, timeout=TMO)
    await b.close()
async def main():
    t = time.time()
    async with async_playwright() as p:
        await asyncio.gather(*[one(p) for _ in range(N)])
    print(f"[preflight]   {N}-way ok in {time.time()-t:.1f}s", flush=True)
asyncio.run(main())
PYEOF
}

# A slow COLD launch predicts rollout aborts even when the warm 8-way is fast:
# rollout spawns browsers in fresh local_process workers that pay cold cost
# again, and env_server /reset gives up at 180s. Observed cold times: g3113 3.1s,
# g3061 10.9s, g3109 138.4s -- and g3109 then aborted 8/8 rollout tasks with
# "BrowserType.launch: Timeout 180000ms exceeded". So gate on the cold time too.
# Threshold is a judgement call, not a measured cliff. Known-good: 3s, 5s, 11s.
# Known-bad: 138s, 213s (aborted every rollout task at env_server's 180s cap).
# 45s rejected g3075 at 46s, which cost a requeue -- and in a contended queue a
# requeue costs hours while a marginal node costs one run. Loosened to 90s.
COLD_MAX="${PREFLIGHT_COLD_MAX_S:-90}"
echo "[preflight] warm-up (1 browser) on $(hostname)..."
_t0=$(date +%s)
if ! run 1 240000 300; then
  echo "[preflight] FATAL: a single browser cannot launch on $(hostname)"
  exit 1
fi
_cold=$(( $(date +%s) - _t0 ))
echo "[preflight] cold launch took ${_cold}s (reject above ${COLD_MAX}s)"
if [ "$_cold" -gt "$COLD_MAX" ]; then
  echo "[preflight] FATAL: cold browser launch ${_cold}s on $(hostname) exceeds ${COLD_MAX}s;"
  echo "[preflight]        this node's I/O will time out rollout browsers (180s cap). Resubmit."
  exit 1
fi
for a in 1 2 3; do
  if run "$N" 180000 240; then
    echo "[preflight] OK (${N}-way, attempt $a) on $(hostname)"
    exit 0
  fi
  echo "[preflight] ${N}-way attempt $a failed on $(hostname); retrying in 20s"
  sleep 20
done
echo "[preflight] FATAL: ${N}-way chromium failed 3x on $(hostname)"
exit 1
