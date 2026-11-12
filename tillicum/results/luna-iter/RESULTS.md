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

## Full fine-tuning (2026-11-12 02:39, pass night-fullft, paired within pass, valid-only)
| model | OM2W | DeepShop |
|---|---|---|
| SFT | 36.1 | 36.5 |
| LoRA r2-luna-r1 | 32.6 | 32.9 |
| full FT lr 1e-6 | **39.3** | **40.9** |
| full FT lr 5e-6 | 35.1 | 32.6 |

full FT 1e-6 - LoRA: OM2W +5.1 [-0.9, +11.1]; DeepShop +7.6 [-0.7, +15.9]. full FT 1e-6 - SFT: +1.2 / +4.1 (n.s.).
lr 5e-6 drifted in training (held-out chosen-action CE 0.19 -> 0.23-0.28) and is not better than LoRA.

## Eval noise: Luna round 1 vs SFT pooled over three passes
| pass | OM2W r1-SFT | DeepShop r1-SFT |
|---|---|---|
| luna-iter | +5.6 | +11.2 |
| sol-relabel | +4.2 | +4.1 |
| night-fullft | -3.9 | -4.2 |
| pooled | +1.9 [-1.3, +5.5] | +3.7 [-1.9, +9.3] |
Single passes swing by 5-8 points; repeat passes (repeat1: SFT, r2-luna-r1, C2, full FT 1e-6) launched to pool.
