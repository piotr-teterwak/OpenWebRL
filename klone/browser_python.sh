#!/bin/bash
# Wrapper used as SLIME_BROWSER_LOCAL_PROCESS_PYTHON.
#
# env_server -> playwright -> chromium must NOT inherit the training
# LD_LIBRARY_PATH. With $CONDA_PREFIX/lib on it, chromium resolves 35 shared
# libs from GPFS (~40s cold launch, 0/8 within playwright's 180s cap under
# 8-way concurrency). With it cleared, all 59 resolve from node-local /lib64
# and none are missing -- which is exactly dan29's working setup (their
# launcher exports only PATH, never LD_LIBRARY_PATH).
#
# Safe because conda's python/playwright binaries carry RPATH to
# $CONDA_PREFIX/lib, so they load without LD_LIBRARY_PATH. Verified:
#   env -u LD_LIBRARY_PATH python -c "import playwright"  -> OK
unset LD_LIBRARY_PATH
exec /gscratch/cse/piotrt/envs/slime/bin/python "$@"
