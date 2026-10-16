#!/usr/bin/env python3
"""Train a top-layer scoring adapter plus selection head on teacher-labeled panels (one GPU).

Generation-time state is preserved: each prompt is encoded by the frozen SFT actor (adapter
off) and only candidate tokens pass through the adapted top layers. The head is warm-started
from a frozen-feature head, so training starts exactly at that head's behavior. Checkpoints are
chosen on held-out GPT-5.5 agreement; branch outcomes and independent Luna agreement are
reported only.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SFT = "/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT"
RELEASE = Path("/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/joint-data-inputs-0d83b48/openwebrl_actor")


def read_jsonl(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", type=Path, required=True)
    ap.add_argument("--branch-panels", type=Path, required=True)
    ap.add_argument("--branch-states", type=Path, required=True)
    ap.add_argument("--reference-picks", type=Path, help="independent teacher picks for agreement; optional")
    ap.add_argument("--external-picks", type=Path, action="append", default=[])
    ap.add_argument("--init-head", type=Path, required=True, help="frozen-feature head.pt to warm-start")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default=SFT)
    ap.add_argument("--states", type=Path, default=RELEASE / "states_full.jsonl")
    ap.add_argument("--top-layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--panels-per-state", type=int, default=4)
    ap.add_argument("--accum", type=int, default=8, help="states per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--train-seconds", type=float, default=2.5 * 3600)
    ap.add_argument("--eval-every", type=int, default=100, help="optimizer steps")
    ap.add_argument("--eval-states", type=int, default=100, help="validation states per periodic eval")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke tests)")
    ap.add_argument("--freeze-head-steps", type=int, default=0,
                    help="keep the warm-started head fixed for this many optimizer steps (adapter-only updates)")
    ap.add_argument("--resume", type=Path, help="output directory of an earlier run with a last/ checkpoint")
    ap.add_argument("--joint", action="store_true",
                    help="Kev-style joint top-layer pass with a <decide> token and pointer readout (selection_joint)")
    ap.add_argument("--pointer-lr", type=float, default=1e-3, help="learning rate for the joint pointer, gate and <decide>")
    ap.add_argument("--kev", action="store_true",
                    help="Kev-style decision pass on the actor's weights (selection_kev): packed prompt + panels, "
                         "LoRA on all layers (use --top-layers 36), pure pointer readout, no warm start")
    ap.add_argument("--decide-token", default="<|fim_pad|>")
    ap.add_argument("--pointer-aux", type=float, default=1.0,
                    help="joint mode: weight of an auxiliary loss on the pointer's own logits")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="with --resume: skip training and run only the final evaluations")
    ap.add_argument("--eval-checkpoint", choices=("best", "last"), default="best",
                    help="checkpoint scored by the final evaluations")
    ap.add_argument("--final-val-states", type=int, help="limit final validation (smoke tests)")
    ap.add_argument("--branch-limit", type=int, help="limit branch evaluation (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from openwebrl.selection_adapter import add_scoring_adapter, score_candidates
    from openwebrl.selection_head import CandidateSetHead, HeadConfig, selection_loss, topk_agreement
    from openwebrl.selection_head_features import candidate_tokens, candidate_tokens_from_ids
    from train_selection_head import evaluate_agreement, evaluate_outcomes, plurality_target

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    log = open(args.output / "log.jsonl", "a")
    emit = lambda record: (log.write(json.dumps(record) + "\n"), log.flush(), print(json.dumps(record), flush=True))

    by_state = defaultdict(list)
    for row in read_jsonl(args.panels):
        by_state[row["base_state_id"]].append(row)
    splits = defaultdict(list)
    for base, rows in by_state.items():
        splits[rows[0]["split"]].append(base)
    prompts = {r["state_id"]: r["prompt_text"] for r in read_jsonl(args.states) if r["state_id"] in by_state}
    branch = read_jsonl(args.branch_panels)[:args.branch_limit]
    prompts.update({r["state_id"]: r["prompt_text"] for r in read_jsonl(args.branch_states)})

    checkpoint = torch.load(args.init_head, map_location="cpu")
    config = HeadConfig.from_json(checkpoint["config"])
    layers, pools = config.layers, config.pools
    head = CandidateSetHead(config)
    head.load_state_dict(checkpoint["state_dict"])
    head.to(args.device)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                        device_map=args.device, attn_implementation="sdpa",
                                                        local_files_only=True)
    model = add_scoring_adapter(model, top_layers=args.top_layers, rank=args.rank, alpha=2 * args.rank)
    pad_id = tokenizer.pad_token_id or 0
    scorer = None
    if args.joint and args.kev:
        raise SystemExit("Choose one of --joint and --kev")
    if args.joint:
        from openwebrl.selection_joint import JointScorer
        scorer = JointScorer(model, head, args.top_layers).to(args.device)
        extras = [scorer.decide_offset, scorer.gate, *scorer.q.parameters(), *scorer.k.parameters()]
    elif args.kev:
        from openwebrl.selection_kev import KevStyleScorer
        decide_id = tokenizer.convert_tokens_to_ids(args.decide_token)
        if decide_id is None or decide_id == tokenizer.unk_token_id:
            raise SystemExit("Unknown <decide> token")
        scorer = KevStyleScorer(model, decide_id).to(args.device)
        extras = [scorer.decide_delta, *scorer.q.parameters(), *scorer.k.parameters()]
    else:
        extras = []
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def head_state():
        """Head (and joint pointer) weights only; never the 4B backbone."""
        if scorer is None:
            return head.state_dict()
        return {k: v for k, v in scorer.state_dict().items() if not k.startswith("model.")}

    def load_head_state(state):
        if scorer is None:
            head.load_state_dict(state)
        else:
            missing = scorer.load_state_dict(state, strict=False).missing_keys
            if any(not k.startswith("model.") for k in missing):
                raise ValueError("Head/pointer checkpoint is incomplete")

    def score(base, rows):
        image = Image.open(rows[0]["image"]).convert("RGB")
        prefix = processor(text=[prompts[base]], images=[image], return_tensors="pt").to(args.device)
        cands = [candidate_tokens_from_ids(tokenizer, ids) if "candidate_token_ids" in row
                 else candidate_tokens(tokenizer, text)
                 for row in rows for text, ids in zip(row["candidates"],
                                                       row.get("candidate_token_ids", row["candidates"]))]
        panels = [cands[i:i + 5] for i in range(0, len(cands), 5)]
        if args.kev:
            return scorer(prefix, panels)
        if scorer is not None:
            logits, pointer = scorer(prefix, panels, args.batch_size, pad_id, return_pointer=True)
            score.pointer = pointer
            return logits
        state, feats = score_candidates(model, prefix, cands, layers, pools, args.batch_size, pad_id)
        feats = feats.reshape(len(rows), 5, -1)
        return head(feats, state.reshape(1, -1).expand(len(rows), -1))[0]

    def evaluate_teacher(bases):
        model.eval(); head.eval()
        logits, pointers, targets = [], [], []
        with torch.no_grad():
            for base in bases:
                rows = by_state[base]
                logits.append(score(base, rows).float().cpu())
                if args.joint:
                    pointers.append(score.pointer.float().cpu())
                targets.append(torch.tensor([r["target"] for r in rows]))
        logits, targets = torch.cat(logits), torch.cat(targets)
        informative = (targets > 0).sum(-1) < 5
        model.train(); head.train()
        result = {"panels": len(targets), "agreement": topk_agreement(logits, targets),
                  "agreement_informative": topk_agreement(logits[informative], targets[informative])}
        if args.joint:
            pointers = torch.cat(pointers)
            result.update(pointer_agreement=topk_agreement(pointers, targets), gate=float(scorer.gate))
        return result

    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr},
              {"params": head.parameters(), "lr": args.head_lr}]
    if extras:
        params.append({"params": extras, "lr": args.pointer_lr})
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    train_bases, val_bases = sorted(splits["train"]), sorted(splits["validation"])
    # A dedicated generator keeps the probe identical across runs and resumes.
    val_probe = random.Random(args.seed).sample(val_bases, min(args.eval_states, len(val_bases)))
    step, seen, order = 0, 0, []

    def save_last():
        """Latest state for resuming: adapter, head, optimizer, counters and data order."""
        model.save_pretrained(args.output / "last" / "adapter")
        torch.save({"head": head_state(), "optimizer": opt.state_dict(), "step": step, "seen": seen,
                    "order": order, "rng": rng.getstate(), "best": best}, args.output / "last" / "state.pt")

    if args.resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(args.resume / "last" / "adapter" / "adapter_model.safetensors")))
        saved = torch.load(args.resume / "last" / "state.pt", weights_only=False)
        load_head_state(saved["head"]); opt.load_state_dict(saved["optimizer"])
        step, seen, order, best = saved["step"], saved["seen"], saved["order"], saved["best"]
        rng.setstate(saved["rng"])
        for name in ("head.pt", "adapter"):
            import shutil
            source, dest = args.resume / name, args.output / name
            if source.resolve() != dest.resolve():
                shutil.copytree(source, dest, dirs_exist_ok=True) if source.is_dir() else shutil.copy(source, dest)
        emit({"event": "resumed", "from": str(args.resume), "step": step, "best_step": best["step"]})
    model.train(); head.train()
    if not args.resume:
        best = {"step": 0, **evaluate_teacher(val_probe)}
        emit({"event": "eval", "step": 0, **best})
        torch.save(head_state(), args.output / "head.pt"); model.save_pretrained(args.output / "adapter")
    if args.eval_only and not args.resume:
        raise SystemExit("--eval-only needs --resume")
    started, total = time.time(), 0.
    while (not args.eval_only and time.time() - started < args.train_seconds
           and (args.max_steps is None or step < args.max_steps)):
        if not order:
            order = train_bases[:]; rng.shuffle(order)
        base = order.pop()
        rows = rng.sample(by_state[base], min(args.panels_per_state, len(by_state[base])))
        target = torch.tensor([r["target"] for r in rows], device=args.device)
        logits = score(base, rows)
        every = torch.ones_like(target, dtype=torch.bool)
        loss = selection_loss(logits, target, every)
        if args.joint and args.pointer_aux:
            loss = loss + args.pointer_aux * selection_loss(score.pointer, target, every)
        (loss / args.accum).backward()
        total += loss.item(); seen += 1
        if seen % args.accum:
            continue
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
        warm = min(1.0, (step + 1) / 20)
        head_lr = 0.0 if step < args.freeze_head_steps else args.head_lr
        for group, lr in zip(opt.param_groups, (args.lr, head_lr, args.pointer_lr)):
            group["lr"] = lr * warm
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % args.eval_every == 0:
            result = evaluate_teacher(val_probe)
            emit({"event": "eval", "step": step, "train_loss": total / (args.accum * args.eval_every),
                  "elapsed": round(time.time() - started), **result})
            total = 0.
            if result["agreement"] > best["agreement"]:
                best = {"step": step, **result}
                torch.save(head_state(), args.output / "head.pt"); model.save_pretrained(args.output / "adapter")
            save_last()
    if not args.eval_only:
        save_last()   # also keep the state between periodic checks, so a later run can extend this one

    # Final: the chosen checkpoint on the branch states first (most informative, cheapest), then full
    # validation. Branch states are never used for training or checkpoint selection.
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    if args.eval_checkpoint == "best":
        set_peft_model_state_dict(model, load_file(str(args.output / "adapter" / "adapter_model.safetensors")))
        load_head_state(torch.load(args.output / "head.pt"))
    elif args.eval_only:   # resume already loaded the latest state
        pass
    # (training runs evaluating "last" keep the in-memory final state, which save_last just stored)
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "evaluated_checkpoint": args.eval_checkpoint,
              "best_probe": best, "optimizer_steps": step, "train_states_seen": seen}
    model.eval(); head.eval()
    picks, pointer_picks = {}, {}
    with torch.no_grad():
        for row in branch:
            picks[row["state_id"]] = int(score(row["state_id"], [row]).argmax(-1))
            if args.joint:
                pointer_picks[row["state_id"]] = int(score.pointer.argmax(-1))
    split = {"rows": branch, "plurality": torch.tensor([plurality_target(r["candidates"]) for r in branch])}
    external = {}
    for path in args.external_picks:
        external[path.stem] = {r["state_id"]: r["selected_index"] for r in read_jsonl(path) if r["selected_index"] is not None}
    reference = {r["state_id"]: r["selected_indices"] for r in read_jsonl(args.reference_picks)} if args.reference_picks else {}
    result["branch_outcomes"] = evaluate_outcomes([picks[r["state_id"]] for r in branch], split, external)
    if reference:
        result["branch_reference_agreement"] = evaluate_agreement(
            {"head": picks, "actor_first": {r["state_id"]: 0 for r in branch}, **external}, split, reference)
    result["branch_head_picks"] = picks
    if args.joint:   # the comparison pointer on its own, whatever the gate
        result["branch_pointer_outcomes"] = evaluate_outcomes([pointer_picks[r["state_id"]] for r in branch], split, {})
        if reference:
            result["branch_pointer_reference_agreement"] = evaluate_agreement({"pointer": pointer_picks}, split, reference)
        result["branch_pointer_picks"] = pointer_picks
        emit({"event": "branch_pointer", "outcomes": result["branch_pointer_outcomes"]["all"],
              "reference_agreement": result.get("branch_pointer_reference_agreement")})
    emit({"event": "branch", "outcomes": result["branch_outcomes"]["all"],
          "reference_agreement": result.get("branch_reference_agreement")})
    (args.output / "metrics.json").write_text(json.dumps(result, indent=1))
    if args.final_val_states == 0:   # branch scoring only; this checkpoint's validation already exists
        return
    result["validation"] = evaluate_teacher(val_bases[:args.final_val_states])
    emit({"event": "final_validation", **result["validation"]})
    (args.output / "metrics.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
