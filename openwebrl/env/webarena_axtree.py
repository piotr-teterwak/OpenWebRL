"""WebArena's accessibility-tree observation of the current viewport, for selectors that judge from a tree (WebArbiter).

    tree = await webarena_axtree(page)   # {"text", "nodes", "viewport"} or {"error"}

Port of TextObervationProcessor (observation_type="accessibility_tree", current_viewport_only=True) from the released
WebArbiter code (YaoZ720/WebArbiterCode @ 764ec26, viewer/src/browser_env/processors.py, which is WebArena's) to
Playwright's async API: Accessibility.getFullAXTree, one getBoundingClientRect per node (text nodes through a Range),
nodes with no box / zero size / < 0.6 of their area in the window removed and their children re-parented (the released
in-viewport ratio expression is kept as written: overlap_w * overlap_h / w * h), the DFS that prints
"[nodeId] role 'name' props" (IGNORED_ACTREE_PROPERTIES skipped, empty generic-like nodes dropped), the StaticText
de-duplication pass, and the "Tab i (current): title | ..." header. obs_nodes_info is returned as `nodes`
({id: {backend_id, union_bound [x, y, w, h] in viewport CSS px, text}}), exactly the released metadata.
Differences from the released processor, none of which changes the tree for a given page state:
  * the window is window.innerWidth x innerHeight. Released code reads window.screen.*, which headless Chromium reports as
    800x600 regardless of the viewport; our WebArena reproduction overrides the screen to the viewport for that reason
    (webarena_repro/PROTOCOL.md D15). Using the viewport directly is that same fix
  * the per-node rect calls are issued concurrently (bounded), and their remote objects are released afterwards
  * DOMSnapshot.captureSnapshot is not taken: on the accessibility-tree path the released code uses only the window
    config from fetch_browser_info, never the snapshot itself
Opt-in only: web_env adds it to observations when OPENWEBRL_WEBARENA_AXTREE=1 (set by arm_eval --selector-page-tree).
"""
import asyncio
import re
from typing import Any

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
IGNORED_ACTREE_PROPERTIES = ("focusable", "editable", "readonly", "level", "settable", "multiline", "invalid")
_EMPTY_ROLES = ["generic", "img", "list", "strong", "paragraph", "banner", "navigation", "Section", "LabelText",
                "Legend", "listitem"]
_RECT_JS = """
    function() {
        if (this.nodeType == 3) {
            var range = document.createRange();
            range.selectNode(this);
            var rect = range.getBoundingClientRect().toJSON();
            range.detach();
            return rect;
        } else {
            return this.getBoundingClientRect().toJSON();
        }
    }
"""
_GROUP = "owrl-webarena-axtree"


async def _rect(client, backend_node_id, gate):
    async with gate:
        try:
            obj = await client.send("DOM.resolveNode", {"backendNodeId": int(backend_node_id), "objectGroup": _GROUP})
            resp = await client.send("Runtime.callFunctionOn", {
                "objectId": obj["object"]["objectId"], "functionDeclaration": _RECT_JS, "returnByValue": True})
            v = resp["result"]["value"]
            return [v["x"], v["y"], v["width"], v["height"]]
        except Exception:
            return None  # released: subtype "error" -> union_bound None


def _in_viewport_ratio(x, y, width, height, win_w, win_h):
    ow = max(0, min(x + width, win_w) - max(x, 0))
    oh = max(0, min(y + height, win_h) - max(y, 0))
    return ow * oh / width * height  # sic (released get_element_in_viewport_ratio)


async def _fetch_tree(client, win_w, win_h):
    tree = (await client.send("Accessibility.getFullAXTree", {}))["nodes"]
    seen, nodes = set(), []
    for node in tree:  # a few nodes are repeated in the accessibility tree
        if node["nodeId"] not in seen:
            nodes.append(node); seen.add(node["nodeId"])
    tree = nodes
    cursor = {n["nodeId"]: i for i, n in enumerate(tree)}
    gate = asyncio.Semaphore(64)
    pending = {}
    for node in tree:
        if "backendDOMNodeId" not in node:
            node["union_bound"] = None
        elif node["role"]["value"] == "RootWebArea":
            node["union_bound"] = [0.0, 0.0, 10.0, 10.0]  # always inside the viewport
        else:
            pending[node["nodeId"]] = _rect(client, node["backendDOMNodeId"], gate)
    rects = await asyncio.gather(*pending.values())
    for nid, r in zip(pending, rects):
        tree[cursor[nid]]["union_bound"] = r
    try:
        await client.send("Runtime.releaseObjectGroup", {"objectGroup": _GROUP})
    except Exception:
        pass

    def remove(node):
        nid = node["nodeId"]; parent = node["parentId"]; children = node["childIds"]
        pc = cursor[parent]
        index = tree[pc]["childIds"].index(nid)
        tree[pc]["childIds"].pop(index)
        for child in children:
            tree[pc]["childIds"].insert(index, child); index += 1
        for child in children:
            tree[cursor[child]]["parentId"] = parent
        tree[cursor[nid]]["parentId"] = "[REMOVED]"

    for node in tree:
        if not node["union_bound"]:
            remove(node); continue
        x, y, w, h = node["union_bound"]
        if w == 0 or h == 0:
            remove(node); continue
        if _in_viewport_ratio(float(x), float(y), float(w), float(h), win_w, win_h) < IN_VIEWPORT_RATIO_THRESHOLD:
            remove(node)
    return [n for n in tree if n.get("parentId", "Root") != "[REMOVED]"]


def _parse_tree(tree):
    idx_of = {n["nodeId"]: i for i, n in enumerate(tree)}
    info = {}

    def dfs(idx, obs_id, depth):
        out = ""; node = tree[idx]; indent = "\t" * depth; valid = True
        try:
            role = node["role"]["value"]; name = node["name"]["value"]
            s = f"[{obs_id}] {role} {name!r}"
            props = []
            for p in node.get("properties", []):
                try:
                    if p["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    props.append(f'{p["name"]}: {p["value"]["value"]}')
                except KeyError:
                    pass
            if props:
                s += " " + " ".join(props)
            if not s.strip():
                valid = False
            if not name.strip():
                if not props:
                    if role in _EMPTY_ROLES:
                        valid = False
                elif role in ["listitem"]:
                    valid = False
            if valid:
                out += f"{indent}{s}"
                info[obs_id] = {"backend_id": node["backendDOMNodeId"], "union_bound": node["union_bound"], "text": s}
        except Exception:
            valid = False
        for child in node["childIds"]:
            if child not in idx_of:
                continue
            cs = dfs(idx_of[child], child, depth + 1 if valid else depth)
            if cs.strip():
                if out.strip():
                    out += "\n"
                out += cs
        return out

    return dfs(0, tree[0]["nodeId"], 0), info


def _clean(tree_str):
    lines = []
    for line in tree_str.split("\n"):
        if "statictext" in line.lower():
            m = re.search(r"\[\d+\] StaticText (.+)", line, re.DOTALL)
            if m:
                text = m.group(1)[1:-1]
                if text and all(text not in prev for prev in lines[-3:]):
                    lines.append(line)
        else:
            lines.append(line)
    return "\n".join(lines)


async def webarena_axtree(page) -> dict[str, Any]:
    """The released accessibility-tree observation of `page` (current viewport), or {"error": ...}."""
    client = None
    try:
        pages = page.context.pages
        try:
            titles = []
            for i, p in enumerate(pages):
                t = await p.title()
                titles.append(f"Tab {i} (current): {t}" if p == page else f"Tab {i}: {t}")
            header = " | ".join(titles)
        except Exception:
            header = " | ".join(["Tab {idx}" for _ in pages])  # sic (released fallback)
        win = await page.evaluate("[window.innerWidth, window.innerHeight]")
        client = await page.context.new_cdp_session(page)
        await client.send("Accessibility.enable")
        tree = await _fetch_tree(client, float(win[0]), float(win[1]))
        text, nodes = _parse_tree(tree)
        return {"text": f"{header}\n\n{_clean(text)}", "nodes": nodes, "viewport": win}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        if client is not None:
            try:
                await client.detach()
            except Exception:
                pass
