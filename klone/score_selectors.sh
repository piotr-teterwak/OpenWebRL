#!/bin/bash
# Score every finished selector eval and put them beside the baselines.
PY=/gscratch/cse/piotrt/envs/slime/bin/python
R=/gscratch/krishna/piotrt/openwebrl-eval
cd "$R" || exit 1
printf "%-10s %-10s %8s %10s %8s %8s\n" SELECTOR JOB "v2" "training" ABORTS TASKS
printf -- "------------------------------------------------------------\n"
for d in /gscratch/scrubbed/piotrt/openwebrl_outputs/eval/*sel-*_nucleus_*; do
  [ -d "$d" ] || continue
  j=$(basename "$d" | grep -oE '[0-9]+$')
  s=$(basename "$d" | sed -E 's/.*sel-([a-z]+)_.*/\1/')
  e=$(find "$d" -maxdepth 2 -type d -name "eval_*turn_*" 2>/dev/null | head -1)
  [ -n "$e" ] || continue
  n=$(find "$e" -name 'results_task_*.jsonl' 2>/dev/null | wc -l)
  [ "$n" -lt 250 ] && { printf "%-10s %-10s %8s %10s %8s %8s\n" "$s" "$j" "-" "-" "-" "$n/300"; continue; }
  v2=$($PY openwebrl/compute_eval_success_rate_woaborted_v2.py "$e" 2>/dev/null | grep -ao "success_rate=[0-9.]*%" | head -1 | cut -d= -f2)
  tr=$($PY openwebrl/compute_eval_success_rate_woaborted_v2.py "$e" --abort-mode training 2>/dev/null | grep -ao "success_rate=[0-9.]*%" | head -1 | cut -d= -f2)
  ab=$($PY openwebrl/compute_eval_success_rate_woaborted_v2.py "$e" 2>/dev/null | grep -ao "aborted=[0-9]*" | head -1 | cut -d= -f2)
  printf "%-10s %-10s %8s %10s %8s %8s\n" "$s" "$j" "${v2:--}" "${tr:--}" "${ab:--}" "$n/300"
done
printf -- "------------------------------------------------------------\n"
printf "%-10s %-10s %8s %10s %8s %8s\n" "n=1 nuc" "39383327" "32.1%" "23.9%" "87" "300/300"
printf "%-10s %-10s %8s %10s %8s %8s\n" "n=1 greedy" "39323119" "34.1%" "28.5%" "74" "300/300"
echo
echo "SCC reference ordering: n=1 33.8 < zs 41.5 < scalar 46.3 < ARM 51.1"
