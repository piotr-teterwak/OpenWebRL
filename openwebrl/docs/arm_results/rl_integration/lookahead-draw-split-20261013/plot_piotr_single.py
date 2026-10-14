"""Piotr-matched states (argv[1]: invalid_as_failure | valid_only): (a) gain over the actor's first sample and (b) absolute success,
both with paired 95% task-clustered bands of the difference from the actor (in (b) drawn around actor + difference)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import STYLE  # noqa: E402

D = Path(__file__).resolve().parent
MODE = sys.argv[1] if len(sys.argv) > 1 else "invalid_as_failure"      # or valid_only
r = {int(k): v["pooled"] for k, v in json.loads((D / "curve_piotr.json").read_text())[MODE].items()}
ns = sorted(r)
ACTOR = "Actor first sample"
actor = [r[n][ACTOR][0] for n in ns]
suffix = "" if MODE == "invalid_as_failure" else "_valid_only"
for kind in ("gain", "absolute", "absolute_lines"):           # absolute_lines: no bands
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for k, (label, color, ls) in STYLE.items():
        if k not in r[ns[0]]:
            continue
        if k == ACTOR:
            y = [0] * len(ns) if kind == "gain" else actor
            ax.plot(ns, y, color=color, ls=ls, lw=1.8 if kind == "gain" else 2.2, marker=None if kind == "gain" else "o", ms=4,
                    label=label, zorder=3)
            continue
        d = [r[n]["minus_actor"][k] for n in ns]
        base = [0] * len(ns) if kind == "gain" else actor
        ax.plot(ns, [b + x[0] for b, x in zip(base, d)], color=color, ls=ls, marker="o", ms=4,
                lw=1.3 if k == "Uniform" else 2, label=label)
        if kind != "absolute_lines":
            ax.fill_between(ns, [b + x[1] for b, x in zip(base, d)], [b + x[2] for b, x in zip(base, d)],
                            color=color, alpha=.10, lw=0)
    ax.set_xticks(ns); ax.set_xlabel("Observed draws per action"); ax.grid(alpha=.25)
    ax.set_ylabel("Gain over actor's first sample (pp)" if kind == "gain" else "Success (%)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), fontsize=9, frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(D / f"lookahead_draw_split_piotr_{kind}{suffix}.{ext}", dpi=160)
    plt.close(fig)
