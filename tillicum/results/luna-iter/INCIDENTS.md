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
