#!/usr/bin/env bash
# Fresh Wolfram Alpha eval for all five paper-table systems, same window (2026-11-13, Piotr: "rerun all baselines once on
# wolfram. IF they all fail, we just consider all wolframs failed for all baselines. Otherwise, we do other two."):
#   run 1 now for all five systems; if every episode is invalid, stop (Wolfram = failure for everyone, overall metric);
#   otherwise runs 2 and 3 for all five systems. Per job:
#      T 1.0 / top-p 0.9 / seed 42 like every other run, 3 browsers per job, step 600 s / task 3600 s
#   -> /gpfs/scrubbed/piotrt/owrl/eval/passwolfram3-r{1,2,3}/<label>/webvoyager ; paper_table.py --wolfram3 uses them
# Budget: nothing is submitted once vlm usage reaches $CAP (950).
set -uo pipefail
cd /gpfs/home/piotrt/OpenWebRL
L=/gpfs/scrubbed/piotrt/owrl/eval/wolfram3; mkdir -p $L; CAP=${CAP:-950}
X="--exclude=g001,g008,g011,g013,g015,g021"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
BLOCK='captcha|are you a robot|not a robot|unusual traffic|automated (access|behavio|traffic|queries)|access denied|press (&|and) hold|verify you are (a )?human|blocked due to|request blocked|you have been blocked'
log() { echo "$(date '+%F %T') $*"; }
usage() { hyakusage -a vlm 2>/dev/null | awk -F'│' '/│ piotrt +│ +[0-9.]+ \//{split($3,a,"/"); gsub(/ /,"",a[1]); print a[1]; exit}'; }
budget_ok() { local u=$(usage); [ -z "$u" ] && return 0; python3 -c "import sys; sys.exit(0 if float('$u') < $CAP else 1)" && return 0; log "BUDGET GUARD: \$$u >= \$$CAP"; return 1; }
wait_jobs() { while [ -n "$(squeue -h -j $(tr ' ' ',' <<<"$*") 2>/dev/null)" ]; do sleep 120; done; }
IDX=$(python3 -c "
import json
def tid(r): return str(r.get('task_id') or (r.get('metadata') or {}).get('task_id') or r.get('id'))
print(','.join(str(i) for i,l in enumerate(open('openwebrl/data/eval/webvoyager_fara.jsonl')) if tid(json.loads(l)).startswith('Wolfram')))")
[ $(tr ',' '\n' <<<"$IDX" | wc -l) = 46 ] || { log "FATAL expected 46 Wolfram tasks"; exit 1; }
log "wolfram3 start (cap \$$CAP); 46 tasks; run 1 now, runs 2-3 only if run 1 has any valid episode"
submit_run() {  # submit_run <k>: all five systems, run k
  local k=$1 P=wolfram3-r$1 J=""
  local C="BENCH=webvoyager PASS=$P INDICES=$IDX PARALLEL=3 TEMP=1.0 STEP_TIMEOUT=600 TASK_TIMEOUT=3600"
  for s in sft adm; do J="$J $(env $C KIND=$s sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-$s-r$k tillicum/eval_redo_sftadm.sbatch 2>/dev/null)"; done
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-boq-r$k best_of_q/eval_q.sbatch 2>/dev/null)"
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-procua-r$k pro_cua/eval_procua_sel.sbatch 2>/dev/null)"
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-dreamer-r$k webdreamer/eval_webdreamer.sbatch 2>/dev/null)"
  echo $J
}
count() {  # count <pass>: "valid blocked total" over the five systems
  python3 - "$1" "$BLOCK" <<'PYEOF'
import glob, json, re, sys
E = '/gpfs/scrubbed/piotrt/owrl/eval'; pat = re.compile(sys.argv[2], re.I); v = b = n = 0
for lab in ['sft1', 'piotr5', 'base+boq', 'base+procuasel', 'base+webdreamer']:
    for f in glob.glob(f'{E}/pass{sys.argv[1]}/{lab}/webvoyager/results/*.json'):
        r = json.load(open(f)); n += 1
        if str(r.get('valid')) == 'True':
            v += 1; b += bool(pat.search(str(r.get('response') or '')))
print(v, b, n)
PYEOF
}
budget_ok || exit 1
J=$(submit_run 1); log "run 1 submitted: $J"; wait_jobs $J
read v b n <<<"$(count wolfram3-r1)"; log "run 1: $v valid of $n episodes ($b of the valid mention a bot wall in the final answer)"
for lab in sft1 piotr5 base+boq base+procuasel base+webdreamer; do
  s=/gpfs/scrubbed/piotrt/owrl/eval/passwolfram3-r1/$lab/webvoyager/summary.json; log "  $lab: $(tr -d '\n ' < $s 2>/dev/null || echo 'no summary')"; done
if [ "${v:-0}" -eq 0 ]; then
  log "ALL FAILED: Wolfram counted as failed for every system (paper_table.py --wolfram3, runs 2-3 absent = failures)"
else
  budget_ok || exit 1
  J=$(submit_run 2) ; J="$J $(submit_run 3)"; log "runs 2-3 submitted: $J"; wait_jobs $J
  for k in 2 3; do read v b n <<<"$(count wolfram3-r$k)"; log "run $k: $v valid of $n ($b mention a bot wall)"; done
fi
python3 tillicum/paper_table.py --wolfram3 > tillicum/results/paper-table/wolfram3.txt 2>&1
log "WOLFRAM3_DONE"; cat tillicum/results/paper-table/wolfram3.txt
