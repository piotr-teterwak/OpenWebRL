#!/bin/bash
# Score every selector on the SAME task subset — the intersection of tasks all
# runs completed. scalar's judge is ~20x slower than arm's (HF eager PeftModel
# vs an sglang server), so it will not finish 300 within its time limit; scoring
# the full runs against a partial one would not be a fair comparison.
PY=/gscratch/cse/piotrt/envs/slime/bin/python
R=/gscratch/krishna/piotrt/openwebrl-eval
W=/gscratch/scrubbed/piotrt/openwebrl_outputs/eval/_intersect
rm -rf "$W"; mkdir -p "$W"
declare -A RUN
for d in /gscratch/scrubbed/piotrt/openwebrl_outputs/eval/*sel-*_nucleus_*; do
  [ -d "$d" ] || continue
  s=$(basename "$d" | sed -E 's/.*sel-([a-z]+)_.*/\1/')
  e=$(find "$d" -maxdepth 2 -type d -name "eval_*turn_*" 2>/dev/null | head -1)
  [ -n "$e" ] && RUN[$s]="$e"
done
# task ids present in every run
common=""
for s in "${!RUN[@]}"; do
  ids=$(find "${RUN[$s]}" -name 'results_task_*.jsonl' -printf '%f\n' 2>/dev/null | sort)
  if [ -z "$common" ]; then common="$ids"; else common=$(comm -12 <(echo "$common") <(echo "$ids")); fi
done
n=$(echo "$common" | grep -c .)
echo "selectors: ${!RUN[@]}"
echo "common tasks: $n"
echo
printf "%-9s %8s %10s %8s\n" SELECTOR v2 training TASKS
printf -- "----------------------------------------\n"
for s in "${!RUN[@]}"; do
  sub="$W/$s/OpenWebRL-4B-SFT/eval_sub"; mkdir -p "$sub"
  while read -r fn; do [ -n "$fn" ] && ln -sf "${RUN[$s]}/$fn" "$sub/$fn"; done <<< "$common"
  for extra in $(ls "${RUN[$s]}" 2>/dev/null | grep -v results_task_); do
    ln -sf "${RUN[$s]}/$extra" "$sub/$extra" 2>/dev/null
  done
  cd "$R"
  v2=$($PY openwebrl/compute_eval_success_rate_woaborted_v2.py "$sub" 2>/dev/null | grep -ao "success_rate=[0-9.]*%" | head -1 | cut -d= -f2)
  tr=$($PY openwebrl/compute_eval_success_rate_woaborted_v2.py "$sub" --abort-mode training 2>/dev/null | grep -ao "success_rate=[0-9.]*%" | head -1 | cut -d= -f2)
  printf "%-9s %8s %10s %8s\n" "$s" "${v2:--}" "${tr:--}" "$n"
done
