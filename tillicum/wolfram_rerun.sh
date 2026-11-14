#!/usr/bin/env bash
# Fresh 3-run Wolfram Alpha eval for all five paper-table systems (2026-11-13, Piotr: "run 3 seeds of each model on
# wolfram again ... to put each model on equal footing", overall accuracy). The site blocks the cluster, so this waits:
#   1. every 3 h, one browser-UA request from the login node; proceed when it returns 200 without a bot wall
#   2. a 3-task real-browser check (SFT); proceed only if >= 2 episodes are valid and none show a bot wall
#   3. all 15 jobs at once (5 systems x 3 runs, 46 tasks each, same window so no system gets a better day):
#      T 1.0 / top-p 0.9 / seed 42 like every other run, 3 browsers per job, step 600 s / task 3600 s
#   -> /gpfs/scrubbed/piotrt/owrl/eval/passwolfram3-r{1,2,3}/<label>/webvoyager ; paper_table.py --wolfram3 uses them
# Gives up after 3 days. Budget: nothing is submitted once vlm usage reaches $CAP (950).
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
log "wolfram3 start (cap \$$CAP); 46 tasks"
deadline=$(( $(date +%s) + 3*86400 )); n=0
while :; do
  [ $(date +%s) -lt $deadline ] || { log "GAVE UP: Wolfram still blocked after 3 days"; exit 1; }
  # 1. login-node probe
  r=$(env -u https_proxy -u http_proxy -u HTTPS_PROXY -u HTTP_PROXY curl -s -o $L/probe.html -w '%{http_code}' --max-time 25 -A "$UA" \
      -H 'Accept-Language: en-US,en;q=0.9' -H 'Accept: text/html' "https://www.wolframalpha.com/input?i=2%2B2" || true)
  if [ "$r" != 200 ] || grep -qiE "$BLOCK" $L/probe.html; then log "probe: http $r (blocked)"; sleep 10800; continue; fi
  log "probe: http 200, no bot wall -> real-browser check"
  # 2. real-browser check from a compute node
  budget_ok || exit 1; n=$((n+1)); P=wolfram3-check$n
  j=$(KIND=sft BENCH=webvoyager PASS=$P INDICES=$(cut -d, -f1-3 <<<"$IDX") PARALLEL=3 sbatch --parsable -A vlm $X --time=00:45:00 \
      --job-name=w3-check tillicum/eval_redo_sftadm.sbatch 2>/dev/null); log "check job $j"; wait_jobs $j
  d=/gpfs/scrubbed/piotrt/owrl/eval/pass$P/sft1/webvoyager/results
  v=$(python3 -c "import glob,json; print(sum(str(json.load(open(f)).get('valid'))=='True' for f in glob.glob('$d/*.json')))" 2>/dev/null || echo 0)
  b=$(grep -liE "$BLOCK" $d/*.json 2>/dev/null | wc -l)
  if [ "${v:-0}" -lt 2 ] || [ "$b" -gt 0 ]; then log "check: $v/3 valid, $b with a bot wall -> still blocked"; sleep 10800; continue; fi
  log "check: $v/3 valid, no bot wall -> launching 15 jobs"; break
done
# 3. all 15 jobs together
budget_ok || exit 1; J=""
for k in 1 2 3; do
  P=wolfram3-r$k; C="BENCH=webvoyager PASS=$P INDICES=$IDX PARALLEL=3 TEMP=1.0 STEP_TIMEOUT=600 TASK_TIMEOUT=3600"
  for s in sft adm; do J="$J $(env $C KIND=$s sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-$s-r$k tillicum/eval_redo_sftadm.sbatch 2>/dev/null)"; done
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-boq-r$k best_of_q/eval_q.sbatch 2>/dev/null)"
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-procua-r$k pro_cua/eval_procua_sel.sbatch 2>/dev/null)"
  J="$J $(env $C sbatch --parsable -A vlm $X --time=02:30:00 --job-name=w3-dreamer-r$k webdreamer/eval_webdreamer.sbatch 2>/dev/null)"
done
log "submitted:$J"; wait_jobs $J
python3 tillicum/paper_table.py --wolfram3 > tillicum/results/paper-table/wolfram3.txt 2>&1
log "WOLFRAM3_DONE"; cat tillicum/results/paper-table/wolfram3.txt
