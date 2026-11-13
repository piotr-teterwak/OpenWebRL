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

## Repeat pass (2026-11-12 04:40, pass repeat1) and pooled differences vs SFT (paired within pass, then pooled)
| model | repeat1 OM2W | repeat1 DeepShop |
|---|---|---|
| SFT | 30.9 | 36.9 |
| LoRA r2-luna-r1 | 36.3 | 33.6 |
| C2 r2-sol-r1-all-m0.3 | 37.8 | 35.1 |
| full FT lr 1e-6 | 38.2 | 35.9 |

| pooled vs SFT | passes | OM2W | DeepShop |
|---|---|---|---|
| LoRA r2-luna-r1 | 4 | +2.4 [-0.4, +5.3] | +1.9 [-2.8, +6.6] |
| C2 | 2 | **+7.1 [+3.0, +11.3]** | +2.7 [-3.4, +8.9] |
| full FT lr 1e-6 | 2 | +3.9 [-0.4, +8.1] | +1.0 [-5.8, +7.9] |

In repeat1, C2, full FT and LoRA r1 are within ~1 point of each other on both benchmarks; the earlier
single-pass gaps over LoRA were mostly its weak night-fullft pass. Only C2 vs SFT on OM2W is significant pooled.

## Iterative v2 step 0 (2026-11-12 17:56, passes v2-r1a + v2-r1b, paired within pass, pooled)
r2-luna-v2-r1: Luna-steered half-A rollouts, all valid states (successful + failed), Luna's executed pick as chosen,
no margin, SFT term only on successful-episode pairs; 6,564 pairs (2,587 successful, 3,977 failed); C2's recipe.
| valid-only | SFT | C2 | v2-r1 | v2-r1 - SFT | C2 - SFT | v2-r1 - C2 |
|---|---|---|---|---|---|---|
| OM2W | 33.8 | 41.6 | 38.3 | +4.4 [+0.2, +8.6] | +5.9 [+1.7, +10.3] | -1.9 [-6.5, +2.3] |
| DeepShop | 35.5 | 42.0 | 36.3 | +1.1 [-5.5, +7.4] | +5.8 [-0.7, +12.0] | -5.1 [-12.1, +1.8] |
Pre-set rule (v2-r1 - SFT pooled over both passes and benchmarks, lower bound > 0): +3.25 [-0.39, +6.76] -> step 1 not run.

## C2 vs SFT pooled over all 4 passes (sol-relabel-c2, repeat1, v2-r1a, v2-r1b)
| | C2 - SFT |
|---|---|
| OM2W | **+6.5 [+3.6, +9.5]** n=971 |
| DeepShop | +4.2 [-0.2, +8.6] n=568 |
| both | **+5.7 [+3.2, +8.2]** n=1539 |
Failed-episode states help with Luna labels too (OM2W), but Sol-labelled C2 is ahead, mostly on DeepShop.
A C2 rerun with Sol's plain prompt and no margin (r2-solp-r1-all) is running to separate prompt format / filter.

## C2 with Sol plain prompt, no margin (2026-11-13 04:36, passes c2p-a + c2p-b, paired within pass, pooled)
r2-solp-r1-all: Sol relabelled with the plain prompt (no probability request), no margin, 6,670 pairs (2,588 successful,
4,082 failed, DPO-only on failed), C2's recipe. Plain vs probability-prompt Sol picks agree on 77-79% of states; both agree
with Luna ~63-65%.
| valid-only | SFT | C2 (prob + margin) | C2-plain | C2-plain - SFT | C2 - SFT | C2-plain - C2 |
|---|---|---|---|---|---|---|
| OM2W | 33.7 | 38.7 | 36.7 | +2.2 [-1.8, +6.3] | +4.0 [-0.4, +8.6] | -1.7 [-6.0, +2.5] |
| DeepShop | 31.6 | 37.1 | 30.7 | -0.7 [-6.6, +5.6] | +5.2 [-1.4, +11.8] | -5.3 [-12.0, +1.4] |
| both | | | | +1.1 [-2.1, +4.7] | **+4.4 [+0.6, +8.1]** | -3.0 [-6.7, +0.5] |
The probability prompt and/or the 0.3 margin help once failed-episode states are included (opposite of C0/C1 on
successful-only states). Round 2 (sol_round2.sh) therefore uses C2's recipe (FORCE_RECIPE=prob; the pre-set default
would have chosen plain because C2 - plain is not significant at 95%).
