# luna-iter incidents

## 2026-11-10 07:2x — validation screenshots purged; driver seed bug
- **Validation images gone.** Round-1 training (444304) failed in 32 s: the C2 validation rows (copied from
  `joint-data/matched`) point at `/gpfs/scrubbed/zixianma/.../runs/c2-full-282782-20260908T075414Z/images/`, which
  no longer exists (scrubbed 60-day purge, ~Nov 7). Training rows were unaffected (they use our own collections).
- **Recovered** all 407 screenshots from `zixianma/arm-offline-data` (HF, image_shards), matched by sha256, into
  `/gpfs/scrubbed/piotrt/owrl/zixian-c2-images/images/` under the original filenames; one image is shared with a
  Piotr state under a different name. Validation / panel / manifest paths in `joint-data/{matched,luna-A,luna-A-s42,
  luna-A-s43,luna-Bsft-s42}` repointed there (originals kept as `*.pre-repoint`), `data_hashes` in each
  `train-config.json` recomputed; every image reference resolves and every repointed file matches its recorded sha256.
  Image content is byte-identical, so validation numbers stay comparable. `matched`'s *train* rows still point at the
  purged directory (2,515 rows; not used by this experiment).
- **Driver bug.** `wait_job` did `read s t q` without `local`, overwriting `collect()`'s seed `s`; the control
  shard was resubmitted with `SEED=RUNNING` (443573 failed at config; 445855 cancelled). Fixed (locals), driver
  restarted 07:26; finished stages were skipped. Log prefixers now `sed -u` (pipeline.log was block-buffered).
- **Maintenance** 11-10 09:00 → 11-11 09:00: training resubmitted as 445890, shortened to fit before it.

## 2026-11-10 08:50 — working mirror of Zixian's scrubbed runtime
- Everything else the pipeline used under `/gpfs/scrubbed/zixianma/` (SFT model, selection ARM 81b452d8, both venvs,
  joint-data-inputs-0d83b48, joint-data-v2 working.sqlite, Megatron-LM src, c2-full tasks.jsonl) was also >60 days old.
  At Piotr's request, copied (fresh timestamps) to `/gpfs/scrubbed/piotrt/owrl/zixian-mirror/` with the same layout
  (~36 GB). The ARM venv's `.pth` and the Megatron editable finder point at the mirror; both venvs import
  torch / transformers / peft / sglang from the mirror only.
- `tillicum/{c2_collect,train_joint,gate,eval_concurrent,build_round2_data,build_combined_data,probe_round2}.sbatch`
  and `scripts/h200_env.sh` now use the mirror. `scripts/train_arm_joint_sft.py` is unchanged (its hash is in the
  frozen r2-luna-r1 config; its defaults are overridden by env vars).
- New pair sets run `scripts/repoint_zixian_paths.py` before verification, so inherited paths (Piotr validation
  screenshots) point at the mirror and the recorded hashes cover the rewritten files.
- Jobs resuming frozen state keep Zixian's SFT path (their frozen configs record it): r2-luna-r1 training (resume
  from update 44) and the luna-Bsft-s43 shard 2 collection. Everything new (round 2, joint / continued / control
  training, gates, evals) uses the mirror. If her SFT is purged before those two finish, restart them from scratch on
  the mirror (r1: 44 updates lost).

## 2026-11-10 14:40 — mirror missed the Playwright browsers; session loss
- Maintenance ended ~14:39. r2-luna-r1 resumed from update 44 on the mirror venv (446004) and trains normally.
- The control shard (446003) "completed" in 2 min: `scripts/h200_env.sh` derives `PLAYWRIGHT_BROWSERS_PATH`,
  `CUDA_HOME`, `CPATH` from the runtime root, and the mirror lacked `browsers/`, `cuda/`, `src/python-headers`, so
  every browser launch failed and 445 tasks were recorded as unavailable (0 turns). Copied those dirs (~1.3 GB; only
  deb/rpm dep lists and one extensions dir were unreadable) and test-launched headless Chromium from the mirror.
  The 445 bogus results moved to `luna-Bsft-s43-shard2/quarantine-446003/` (51 earlier results kept); resubmitted.
- The Claude session ended during maintenance, taking the driver / backup / urgent loops with it; restarted 14:55.
  `train()` now adopts a queued/running job of the same run instead of submitting a duplicate.

## 2026-11-10 16:32 — round-1 "GATE FAIL" was the probe, not the model
- r2-luna-r1 finished all 162 updates. Its gate (446383) crashed in 88 s: `probe_round2.sbatch` samples states from
  Zixian's `c2-full-282782` collection, whose `training.jsonl` is not in the mirror and whose screenshots are purged.
  The driver read the crash as a gate fail and stopped round 2.
- Probe states now come from `luna-iter/probe-c2` (our SFT + Luna half-B collection, seed 42, held out from round-1
  training; `PROBE_C2` overrides). Verified 200 states load. Driver restarted 19:08; gate resubmitted (447549).

## 2026-11-11 22:10 — Sol pilot: no Sol recollection
- 100 half-A tasks that SFT+Luna failed in both seeds, rerun with SFT as actor, seed 44: Sol steering 14/92 valid
  (15.2%), Luna steering 22/97 (22.7%). Paired on 89 tasks valid in both: Sol 14, Luna 20 (Sol-only 5, Luna-only 11).
  Pre-set rule was Sol >= Luna + 15 tasks; it is -6. Sol recollection (~$275 OpenAI) skipped.
- Sol run lost 8 tasks to a browser-port race with its Luna twin (same node, same start); fixed for later
  collections (per-job browser port block, 80b144a). Luna API $1.17, Sol API $20.28.
