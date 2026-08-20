#!/usr/bin/env bash
# Keep checkpoint storage bounded: 58G per iteration x 90 iterations would be 5TB.
# Keep the newest KEEP_LAST plus every KEEP_EVERY-th iteration; delete the rest.
# Never touches the iteration named by latest_checkpointed_iteration.txt, and never
# the newest KEEP_LAST, so a checkpoint still being written is always out of scope.
set -u
CKPT="${1:?usage: prune_ckpts.sh <ckpt dir>}"
KEEP_LAST="${KEEP_LAST:-3}"
KEEP_EVERY="${KEEP_EVERY:-10}"

latest_file="$CKPT/latest_checkpointed_iteration.txt"
[ -f "$latest_file" ] || exit 0
latest=$(tr -dc '0-9' < "$latest_file")
[ -n "$latest" ] || exit 0

# iteration numbers present, ascending
mapfile -t nums < <(ls -d "$CKPT"/iter_[0-9]* 2>/dev/null | sed 's|.*/iter_||' | sed 's/^0*//' | sed 's/^$/0/' | sort -n)
[ "${#nums[@]}" -gt 0 ] || exit 0

# the newest KEEP_LAST are protected
protected=" $latest "
start=$(( ${#nums[@]} - KEEP_LAST )); [ "$start" -lt 0 ] && start=0
for ((i=start; i<${#nums[@]}; i++)); do protected+="${nums[$i]} "; done

for n in "${nums[@]}"; do
  case "$protected" in *" $n "*) continue ;; esac
  [ $(( n % KEEP_EVERY )) -eq 0 ] && continue
  # An hf_converted/ inside the iteration means an eval job converted it and may be
  # serving it right now; deleting it would kill that job mid-run.
  [ -d "$CKPT/iter_$(printf '%07d' "$n")/hf_converted" ] && { echo "[prune_ckpts] keeping iter_$(printf '%07d' "$n") (has hf_converted)"; continue; }
  d="$CKPT/iter_$(printf '%07d' "$n")"
  [ -d "$d" ] && rm -rf "$d" && echo "[prune_ckpts] deleted iter_$(printf '%07d' "$n")"
done
exit 0
