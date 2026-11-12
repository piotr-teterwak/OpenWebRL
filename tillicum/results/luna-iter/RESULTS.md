# luna-iter / Sol relabel / DAgger results (valid-only success, T 1.0, one eval run per model)

## C2: failed-episode states help (2026-11-12 00:33, pass sol-relabel-c2, paired within pass)
| model | OM2W | DeepShop |
|---|---|---|
| SFT | 31.7 | 33.8 |
| C1 r2-sol-r1-m0.3 (Sol labels, margin 0.3, successful episodes) | 34.6 | 31.7 |
| C2 r2-sol-r1-all-m0.3 (C1 + Sol-labelled failed-episode states, DPO-only on those) | **41.2** | **42.2** |

C2 - C1: OM2W +6.9 [+0.8, +13.4] n=247; DeepShop +9.8 [+0.7, +18.2] n=143.
C2 - SFT: OM2W +9.2 [+3.6, +15.1]; DeepShop +7.6 [-1.4, +16.6].
Caveat: C2 has 5,304 pairs vs C1's 2,044 (data size not controlled), but the luna-iter control (2x successful-episode
data) did not help, so the failed-episode states are the likely cause.

## Earlier (see INCIDENTS.md and the conversation summary for the full tables)
- Luna round 1 (r2-luna-r1) is +4-6 over SFT; control (2x data), round 2 (joint/continued), epoch-1 checkpoints,
  C0 (Sol labels) and C1 (margin filter) do not beat it.
- Sol pilot: Sol does not steer twice-failed tasks better than Luna (15.2% vs 22.7%); no Sol recollection.
