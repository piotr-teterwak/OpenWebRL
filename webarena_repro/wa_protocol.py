"""Protocol helpers copied verbatim from Zixian Ma's harness (zixianma/OpenWebRL @ arm 763753a):
scripts/run_native_webarena_cell.py (normalize_trajectory, strip_json_fence, validate_single_action_response,
v8_* terminal/stop-pool adapter) and openwebrl/webprm_webarena_worker.py (single_action_system_prompt, envelope v2).
Only imports and the EvaluationNotReady exception are local. See PROTOCOL.md for which ones each method uses.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import re


class EvaluationNotReady(RuntimeError):
    pass


LEGACY_NO_SEARCH_PROMPT = 'single_action_reconstruction_v1'
ENVELOPE_NO_SEARCH_PROMPT = 'single_blocks_envelope_v2'


def normalize_trajectory(raw, action_types):
    """Lossless evaluator projection, without inventing a terminal action.

    The original coordinator ends in the terminal control action without calling
    the browser or appending a new state. Reject any post-terminal state/action;
    the prospective runner must preserve this boundary without trimming evidence.
    """
    result=[]
    for index,entry in enumerate(raw):
        if index%2==0:
            obs=entry['observation'];info=entry['info']
            result.append(dict(observation={k:v for k,v in obs.items() if k!='image'},
                info=dict(fail_error=info.get('fail_error',''),observation_metadata=deepcopy(info['observation_metadata']))))
        else:
            action={k:(v.tolist() if hasattr(v,'tolist') else v) for k,v in entry.items()}
            kind=action_types(int(action['action_type']))
            action['action_type']='send_msg_to_user' if kind==action_types.SEND_MESSAGE_TO_USER else kind.name.lower()
            result.append(action)
    if any(index%2 and entry.get('action_type')=='send_msg_to_user' and index!=len(result)-1
           for index,entry in enumerate(result)):
        raise EvaluationNotReady('Original terminal control cannot have a following browser state or action')
    if result and len(result)%2==0 and result[-1].get('action_type')!='send_msg_to_user':
        raise EvaluationNotReady('Nonterminal browser action lacks its actual following state')
    return result


# The closing fence is optional: ~1% of gpt-4o policy replies open ```json and never close it, and the released
# json.loads parser then yields no candidates, which ends the episode on the failure terminal (found in step 1).
JSON_FENCE = __import__('re').compile(r'\A\s*```(?:json)?[ \t]*\n?(.*?)\n?\s*(?:```\s*)?\Z', __import__('re').S)


def strip_json_fence(text):
    """v7: return the body of a whole-reply ```json fence; any other text is unchanged."""
    match = JSON_FENCE.match(text) if isinstance(text, str) else None
    return match.group(1) if match else text


def validate_single_action_response(text):
    parsed=json.loads(text)
    if not isinstance(parsed,dict) or not isinstance(parsed.get('BLOCKS'),list) or len(parsed['BLOCKS'])!=1:
        raise EvaluationNotReady('Single-action control requires exactly one raw BLOCK')


V8_TERMINAL = re.compile(r'\A\s*send_msg_to_user\b(.*)\Z', re.S)
V8_BRACKETED = re.compile(r'\A\s*send_msg_to_user\s*\[.*\]\s*\Z', re.S)


def v8_bracket(action):
    """v8: execute a bracketless terminal message as `send_msg_to_user [text]`; every other string is unchanged."""
    if not isinstance(action, str):
        return action
    match = V8_TERMINAL.match(action)
    if match is None or V8_BRACKETED.match(action):
        return action
    text = match.group(1).strip()
    return f'send_msg_to_user [{text}]' if text else action


def v8_terminal(action):
    """Bracket-normalize a detector terminal BLOCK; None/empty actions pass through unchanged."""
    if isinstance(action, dict) and isinstance(action.get('action'), str):
        return dict(action, action=v8_bracket(action['action']))
    return action


def v8_plain(value):
    """JSON form of released dataclass values (the released tournament returns a BlockInfo winner)."""
    from dataclasses import is_dataclass
    if is_dataclass(value):
        return v8_plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): v8_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [v8_plain(v) for v in value]
    return value


def v8_action(selected):
    return selected.get('action') if isinstance(selected, dict) else getattr(selected, 'action', None)


def v8_pool(candidates, terminal):
    """Policy candidates plus the detector stop action (unless an identical action is already a candidate)."""
    pool = deepcopy(list(candidates))
    if not terminal or not isinstance(terminal.get('action'), str) or not terminal['action'].strip():
        return pool, None
    for index, candidate in enumerate(pool):
        if candidate.get('action') == terminal['action']:
            return pool, index
    pool.append(dict(thought=terminal.get('thought', ''), action=terminal['action'],
                     confidence=terminal.get('confidence', 0.0), status='v8_terminal_detector_candidate',
                     reasoning=None, meaning=None))
    return pool, len(pool)-1


def v8_decision_adapter(method, select_control, record):
    """Run the released CoordinatorAgent.execute_step with the v8 terminal policy.

    arbiter7b/shepherd8b: the released terminal detection still runs, but its action is held back and added to
    the tournament / WebShepherd pool instead of short-circuiting it. If the policy produced no candidates the
    stop action is the only option and is selected. no_search: unchanged released override (gpt-4o detector).
    select_control(pool, inputs, context) -> (index, trace) for the non-tournament methods.
    """
    async def adapter(*, native, agent, context, inputs):
        stash = {}
        released_detect = type(agent)._detect_terminal_if_needed
        async def detect(*args, **kwargs):
            if __import__('os').environ.get('WA_DETECTOR') == 'off' and method not in ('no_search', 'piotr_selectionarm'):
                return None  # A15: no terminal detector; the episode ends only on a policy-proposed stop or max steps
            action = v8_terminal(await released_detect(agent, *args, **kwargs))
            if __import__('os').environ.get('WA_DETECTOR') == 'released':
                return action  # A16: released behaviour as shipped — step-0-only schedule, and a detected stop short-circuits the tournament
            if method in ('no_search', 'piotr_selectionarm') or not action:  # v10: Piotr keeps the released override
                return action
            stash['terminal'] = deepcopy(action)
            return None
        released_tournament = native.coordinator.run_tournament
        async def tournament(**kwargs):
            pool, stop_index = v8_pool(kwargs['candidates'], stash.get('terminal'))
            if method == 'arbiter7b':
                outcome = await released_tournament(**dict(kwargs, candidates=pool))
                record(inputs['current_step'], dict(method=method, candidates=pool, v8_stop_index=stop_index,
                    v8_terminal=stash.get('terminal'), tournament_success=bool(outcome.get('success')),
                    final_winner=v8_plain(outcome.get('final_winner')) if outcome.get('success') else None,
                    adaptation='v8_stop_competes_in_released_tournament'))
                return outcome
            index, trace = await select_control(pool, inputs, context)
            record(inputs['current_step'], dict(method=method, candidates=pool, selected_index=index, trace=trace,
                v8_stop_index=stop_index, v8_terminal=stash.get('terminal'),
                adaptation='shared_released_policy_matched_control' if method == 'no_search' else
                    'v10_piotr_selectionarm_released_five_block_pool' if method == 'piotr_selectionarm' else 'v8_stop_competes_with_shepherd_pool'))
            return dict(success=True, final_winner=deepcopy(pool[index]))
        agent._detect_terminal_if_needed = detect
        native.coordinator.run_tournament = tournament
        try:
            decision = await agent.execute_step(**inputs)
        finally:
            native.coordinator.run_tournament = released_tournament
            del agent._detect_terminal_if_needed
        if method != 'no_search' and not v8_action(decision.get('selected_action')) and stash.get('terminal'):
            decision = dict(decision, selected_action=deepcopy(stash['terminal']), v8_only_stop_candidate=True)
        return decision
    return adapter


def single_action_system_prompt(original, *, version=LEGACY_NO_SEARCH_PROMPT):
    """Disclosed NoSearch reconstruction with an explicit envelope clarification.

    V1 preserves the original examples. Opt-in V2 wraps the positive example
    in the required BLOCKS array and clarifies the matching instruction.
    Current-state reasoning, action grammar, user prompt, state difference and
    terminal detection remain shared. Neither version is paper-exact evidence.
    """
    replacements = [
        ('You are an autonomous web navigation agent. Generate 5 diverse, PARALLEL action candidates for the current page state.\n'
         'CRITICAL: These are 5 independent choices, NOT a sequential plan. Each candidate must be executable from the CURRENT state.\n\n',
         'You are an autonomous web navigation agent. Generate exactly ONE action for the current page state.\n'
         'CRITICAL: Return one action, NOT a sequential plan. The action must be executable from the CURRENT state.\n\n'),
        ('{"BLOCKS":[{"thought":"...","action":"...","confidence":0.0-1.0}, ... (5 items)]}',
         '{"BLOCKS":[{"thought":"...","action":"...","confidence":0.0-1.0}]}'),
    ]
    prompt = original
    for previous, replacement in replacements:
        if prompt.count(previous) != 1:
            raise EvaluationNotReady('Pinned native single-action adaptation anchor changed')
        prompt = prompt.replace(previous, replacement, 1)
    begin, end = '# DIVERSITY REQUIREMENT CRITICAL\n', '# ACTION RULES\n'
    if prompt.count(begin) != 1 or prompt.count(end) != 1:
        raise EvaluationNotReady('Pinned native diversity-section anchors changed')
    prefix, rest = prompt.split(begin, 1)
    _, suffix = rest.split(end, 1)
    if version == ENVELOPE_NO_SEARCH_PROMPT:
        good_header = '# EXAMPLE (GOOD - with clear justification)\n'
        bad_header = '# ANTI-EXAMPLES (BAD - DO NOT IMITATE)\n'
        if suffix.count(good_header) != 1 or suffix.count(bad_header) != 1:
            raise EvaluationNotReady('Pinned native NoSearch example anchors changed')
        before, example = suffix.split(good_header, 1)
        positive, after = example.split(bad_header, 1)
        lines = positive.splitlines(keepends=True)
        object_lines = [i for i, line in enumerate(lines) if line.startswith('{"thought":')]
        if len(object_lines) != 1:
            raise EvaluationNotReady('Pinned native NoSearch positive example changed')
        index = object_lines[0]
        block = json.loads(lines[index])
        if set(block) != {'thought', 'action', 'confidence'}:
            raise EvaluationNotReady('Pinned native NoSearch example fields changed')
        lines[index] = json.dumps({'BLOCKS': [block]}, separators=(',', ':')) + '\n'
        suffix = before + good_header + ''.join(lines) + bad_header + after
        instruction = ('Return one JSON object with the key BLOCKS whose value is a one-element array.\n'
            'Put thought, action, and confidence inside that array element; never return a bare action object.\n'
            'The single BLOCK represents one action from the CURRENT state.\n')
    else:
        instruction = 'Return exactly one BLOCK representing one action from the CURRENT state.\n'
    return prefix + ('# SINGLE ACTION REQUIREMENT\n'
        + instruction +
        'Do NOT chain actions or return alternative candidates.\n'
        'The action must be executable RIGHT NOW from the current page.\n\n') + end + suffix


# --- A11: WebArbiter's judge prompt in its training (WebPRMBench / BrowserGym) format -------------------------------
# Only the reward prompt WebArbiter reads is rewritten; the policy, the helpers and gpt-4o-mini see the original. The
# verdict is "Response 1/2", so nothing maps back. Measured on labelled WebPRMBench pairs, the WebArena tree style costs
# ~2 points (C4) and the extra sections move ~5% of verdicts (arbiter_sensitivity.py).
_BG_SECTION = re.compile(r'^#### (.+?) ####\s*$', re.M)
_BG_NODE = re.compile(r"""^(\t*)(\[\w+\] \S+ )('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")(.*)$""")
_BG_PROP = re.compile(r' (\w+): (.*?)(?= \w+: |$)')
_BG_STEP = re.compile(r'Step \d+:\nTHOUGHT: (.*?)\nACTION: (.*?)(?=\nStep \d+:|\Z)', re.S)
_BG_ACTION = re.compile(r'^\s*(\w+)((?: \[[^\]]*\])*)')


def browsergym_action(text):
    """WebArena bracket action -> BrowserGym call (click [5] -> click('5'), type [5] [x] [1] -> fill('5', 'x'))."""
    t = re.match(r"^\s*type\s+\[([^\]]+)\]\s*(.*?)\s*$", text or '', re.S)
    if t:  # the released policy writes `type [id] 'text'`, `type [id] text` or `type [id] [text] [0|1]`
        rest = re.sub(r'\s*\[[01]\]$', '', t.group(2))
        rest = rest[1:-1] if len(rest) >= 2 and rest[0] + rest[-1] in ("[]", "''", '""') else rest
        return f'fill({t.group(1)!r}, {rest!r})'
    m = _BG_ACTION.match(text or '')
    if not m:
        return (text or '').strip()
    name, args = m.group(1), re.findall(r'\[([^\]]*)\]', m.group(2))
    q = lambda s: repr(str(s))
    if name in ('click', 'hover') and args:
        return f'{name}({q(args[0])})'
    if name == 'type' and len(args) >= 2:
        return f'fill({q(args[0])}, {q(args[1])})'
    if name == 'press' and args:
        return f'press({q(args[-1])})'
    if name == 'scroll':
        return 'scroll(0, -500)' if args and args[0].strip().lower() == 'up' else 'scroll(0, 500)'
    if name in ('send_msg_to_user', 'stop'):
        return f'send_msg_to_user({q(args[0] if args else "")})'
    if name == 'goto' and args:
        return f'goto({q(args[0])})'
    if name == 'tab_focus' and args:
        return f'tab_focus({args[0]})'
    if name == 'select_option' and len(args) >= 2:
        return f'select_option({q(args[0])}, {q(args[1])})'
    if name == 'report_infeasible':
        return f'report_infeasible({q(args[0] if args else "")})'
    if name == 'clear' and args:
        return f'clear({q(args[0])})'
    if name == 'noop':
        return 'noop()'
    return {'go_back': 'go_back()', 'go_forward': 'go_forward()', 'new_tab': 'new_tab()', 'close_tab': 'tab_close()'}.get(name, text.strip())


def _bg_tree(section):
    head, *lines = section.rstrip('\n').split('\n')
    out, root = [head], None
    for line in lines:
        if not line.strip() or re.match(r'^Tab \d+( \(current\))?:', line):
            continue
        if line.startswith('Note:'):
            out.append(line); continue
        m = _BG_NODE.match(line)
        if m and ' RootWebArea ' in m.group(2) and root is None:
            root = len(m.group(1)); continue
        level = len(line) - len(line.lstrip('\t'))
        indent = '\t' * (level - 1 if root is not None and level > root else level)
        if not m:
            out.append(indent + line.strip()); continue
        props = []
        for k, v in _BG_PROP.findall(m.group(4)):
            if k == 'url':
                continue
            v = v.strip()
            props.append(f'{k}={v}' if v in ('True', 'False') or re.fullmatch(r'-?\d+(\.\d+)?', v) else f'{k}={v!r}')
        out.append(indent + m.group(2) + m.group(3) + ''.join(', ' + p for p in props))
    return '\n'.join(out) + '\n\n'


def _bg_trajectory(section):
    head, _, rest = section.partition('\n')
    note, _, body = rest.partition('\n') if rest.startswith('Note:') else ('', '', rest)
    steps = [dict(thought=t.strip(), action=browsergym_action(re.sub(r' "\[.*$', '', a.strip(), flags=re.S)))
             for t, a in _BG_STEP.findall(body)]
    return '\n'.join(x for x in (head, note) if x) + '\n' + json.dumps(steps, indent=2, ensure_ascii=False) + '\n\n'


def _bg_responses(section):
    return re.sub(r'(ACTION:\n)(.*?)(\n)', lambda m: m.group(1) + browsergym_action(m.group(2)) + m.group(3), section)


def to_browsergym(user_text):
    """Reward-prompt user text -> WebPRMBench layout: no Prior Knowledge / State Difference sections, BrowserGym
    AXTree attributes (no link URLs, no tab header or RootWebArea), JSON trajectory and BrowserGym actions."""
    marks = list(_BG_SECTION.finditer(user_text))
    if not marks:
        return user_text
    parts = [user_text[:marks[0].start()]]
    for m, nxt in zip(marks, marks[1:] + [None]):
        name, text = m.group(1), user_text[m.start():nxt.start() if nxt else len(user_text)]
        if name in ('Prior Knowledge', 'State Difference Summary'):
            continue
        parts.append(_bg_tree(text) if name == 'AXTREE' else _bg_trajectory(text) if name == 'Trajectory'
                     else _bg_responses(text) if name == 'Assistant Responses' else text)
    return ''.join(parts)


# --- A12: the policy may propose the answer -----------------------------------------------------------------------
# The released sample_p.py action space has no stop action, so the released policy can never propose
# `send_msg_to_user`; episodes end only through the gpt-4o-mini terminal detector or the 30-step N/A fallback. The
# authors' own demo episodes (yaozhang.ai/webarbiter/demo, explorer-data.js) show `send_msg_to_user [answer]` among the
# policy's 5 candidates, winning the tournament (Map step 2, Reddit step 4), so the prompt that produced Table 5 had
# one. Its wording is not released; this follows WebArena's original `stop [answer]` description.
POLICY_STOP_BLOCK = (
    "Completion Action:\n"
    "• send_msg_to_user [answer] - Finish the task. Use it when the objective is complete: if the task asks for "
    "information, put the answer read from the current page in the brackets; if the task asks for a change, use it "
    "once the change is done. If you believe the task is impossible, use send_msg_to_user [N/A].\n"
    "\n")


def policy_stop_system_prompt(system_text):
    marker = 'Tab Management Actions:\n'
    if marker not in system_text or 'send_msg_to_user [answer]' in system_text:
        return system_text
    return system_text.replace(marker, POLICY_STOP_BLOCK + marker, 1)


# --- A13: type content without wrapper quotes/brackets ------------------------------------------------------------
# The released prompt writes `type [id] [content]` and the released parser types everything after the id verbatim
# (plus Enter), so GPT-4o's usual `type [id] 'content'` puts literal quote characters into the field (76% of our
# type actions; GPT-4o-mini 26%; the authors' demo policy 0 of 12). Strip one pair of wrapping quotes or brackets in
# the policy reply, so judges and executor both see the bare content the authors' candidates had.
_TYPE_WRAPPED = re.compile(r'''("action"\s*:\s*")(type\s*\[[^\]]+\]\s*)(?:'([^'\\]*)'|\\"((?:[^"\\]|\\[^"])*)\\"|\[([^\]\\]*)\])(\s*")''')


def unwrap_type_content(reply):
    def fix(m):
        content = next(g for g in (m.group(3), m.group(4), m.group(5)) if g is not None)
        return m.group(1) + m.group(2) + content + m.group(6)
    return _TYPE_WRAPPED.sub(fix, reply) if isinstance(reply, str) else reply


# --- A12 "full": the whole action vocabulary WebArbiter was trained on ------------------------------------------------
# WebArbiter-Data (ZYao720, 57k actions, 100% BrowserGym syntax): click 59%, scroll 18%, fill 9%, send_msg_to_user 6.8%,
# go_back/goto/hover/press/new_tab/tab_focus/tab_close/go_forward/clear ~5%, select_option 1.2%, noop 0.3%,
# report_infeasible 0.1%, plus ~60 hallucinated verbs that only appear as losers. The released prompt lacks the stop,
# report_infeasible, select_option, noop and clear; the released id-based parser lacks the first four of those.
POLICY_FULL_BLOCK = (
    "Completion Actions:\n"
    "• send_msg_to_user [answer] - Finish the task. Use it when the objective is complete: if the task asks for "
    "information, put the answer read from the current page in the brackets; if the task asks for a change, use it "
    "once the change is done.\n"
    "• report_infeasible [reason] - Finish the task by telling the user it cannot be done as instructed (the "
    "requested item, option or permission does not exist). This ends the task.\n"
    "\n"
    "Other Page Actions:\n"
    "• select_option [id] [option] - Choose an option in a dropdown/select element with id, by the option's visible text\n"
    "• clear [id] - Clear the text in an input field with id\n"
    "• noop - Do nothing this step (wait for the page to finish loading)\n"
    "\n")


def policy_actions_system_prompt(system_text, mode):
    """mode 'stop': only send_msg_to_user (POLICY_STOP_BLOCK); mode 'full': POLICY_FULL_BLOCK."""
    block = POLICY_FULL_BLOCK if mode == 'full' else POLICY_STOP_BLOCK
    marker = 'Tab Management Actions:\n'
    if marker not in system_text or 'send_msg_to_user [answer]' in system_text:
        return system_text
    return system_text.replace(marker, block + marker, 1)


_EXTRA_ACTION = re.compile(r'^\s*(report_infeasible|select_option|noop)\b\s*(.*)$', re.S)


def parse_extra_action(action_str, actions):
    """Released id-based parser extension: returns a released Action for the three verbs it lacks, else None.
    report_infeasible [reason] -> send_msg_to_user with answer "N/A" (WebArena grades infeasible tasks by that answer;
    BrowserGym's report_infeasible is scored the same way); select_option [id] [option] -> released
    SELECT_DROPDOWN_OPTION (clicks the element, matches the option's visible text); noop -> released NONE action."""
    m = _EXTRA_ACTION.match(action_str or '')
    if not m:
        return None
    verb, rest = m.group(1), m.group(2).strip()
    if verb == 'noop':
        return actions.create_none_action()
    if verb == 'report_infeasible':
        a = actions.create_send_message_to_user_action('N/A'); a['raw_prediction'] = action_str; return a
    ids = re.findall(r'\[([^\]]*)\]', rest)
    if len(ids) < 2:
        raise actions.ActionParsingError(f'Invalid select_option action {action_str}: expected select_option [id] [option]')
    a = actions.create_select_dropdown_option_action(argument=ids[1].strip(), element_id=ids[0].strip()); a['raw_prediction'] = action_str
    return a


# --- A14: the judge prompt WebArbiter was trained on (paper Appendix D) ---------------------------------------------
# WebArbiter-Data (SFT 9,642 + RL 18,921): system = the two sentences below, no "#### Evaluation Criteria ####" block,
# user = Intent / AXTREE / Trajectory / start url / current url (with the URL hint) / Assistant Responses / Output
# Instructions. The released reward_p.py adds the fixed five-criteria block, a "[General Evaluation Criteria]" intro and
# the Prior Knowledge + State Difference sections; WebPRMBench prompts also carry the criteria block.
TRAINING_JUDGE_SYSTEM = ("You are a skilled expert at evaluating assistant responses. You should evaluate given responses based on "
                         "the judging criteria.\n Given the context of the conversation and two responses from the Assistant, you "
                         "need to refer to determine the better response. Provide an overall comprehensive comparison upon them.\n")


def training_judge_prompt(system_text, user_text):
    """(system, user) in the training format: training system text; user without the criteria block (if the released
    build put it there) and without the Prior Knowledge / State Difference sections. Action and tree syntax unchanged."""
    user = strip_extra(user_text) if 'strip_extra' in globals() else user_text
    user = re.sub(r'#### Evaluation Criteria ####.*?(?=#### Intent ####)', '', user, flags=re.S)
    return TRAINING_JUDGE_SYSTEM, user


def strip_extra(user_text):
    marks = list(_BG_SECTION.finditer(user_text))
    if not marks:
        return user_text
    out = [user_text[:marks[0].start()]]
    for m, nxt in zip(marks, marks[1:] + [None]):
        if m.group(1) not in ('Prior Knowledge', 'State Difference Summary'):
            out.append(user_text[m.start():nxt.start() if nxt else len(user_text)])
    return ''.join(out)


# --- A17: Web-Shepherd candidate generation (arXiv 2505.15277 App. F.2 / Sec. 6.2) --------------------------------------
# "we sample 20 output sequences using nucleus sampling with a temperature of 1.0. The top-n most frequent actions across
# these samples are selected as candidates ... In cases where multiple actions receive the same score, we execute the
# action that was sampled more frequently"; Sec. 6.2: "Among 5 action candidates sampled from the policy". The samples
# come from one chat call with n=20 (wa_episode.shepherd_sample); this function turns them into one released-format reply.
# WA_SAMPLES overrides the sample count (A19: 5 independent samples, all distinct ones kept); default = Web-Shepherd's 20.
SHEPHERD_SAMPLES, SHEPHERD_KEEP = int(__import__('os').environ.get('WA_SAMPLES') or 20), 5


def shepherd_candidates(replies, k=SHEPHERD_KEEP, n=SHEPHERD_SAMPLES):
    """Aggregate n single-BLOCK policy replies into one released-format BLOCKS reply (JSON string) plus statistics.

    Each reply is read like the released PolicyGenerator._parse_response (whole-reply fence stripped, {"BLOCKS": [...]},
    thought and action non-empty; the first BLOCK of a reply counts). Actions are grouped by exact string, as the
    released dedup does. The k most frequent (ties: first sampled) become the BLOCKS, most frequent first, with
    confidence = frequency / n so the released confidence rerank keeps that order (the Web-Shepherd tie-break), the
    count in the released `frequency` field, and the thought and stated confidence of the first sample of each action.
    """
    groups, order, dropped = {}, [], 0
    for reply in replies:
        try:
            parsed = json.loads(strip_json_fence(reply))
            blocks = parsed.get('BLOCKS') if isinstance(parsed, dict) else None
            block = blocks[0] if isinstance(blocks, list) and blocks and isinstance(blocks[0], dict) else {}
            thought, action = (block.get('thought') or '').strip(), (block.get('action') or '').strip()
        except Exception:
            thought = action = ''
        if not thought or not action:
            dropped += 1; continue
        if action not in groups:
            groups[action] = dict(thought=thought, action=action, stated_confidence=block.get('confidence'), frequency=0)
            order.append(action)
        groups[action]['frequency'] += 1
    if __import__('os').environ.get('WA_MERGE_STOP') == '1':
        # A20: every send_msg_to_user / report_infeasible sample is one "stop" vote. The stop enters as a single
        # candidate with the summed frequency, worded as its most frequent stop (ties: first sampled), so reworded
        # stops no longer take several of the k slots each with frequency 1 (A18 CMS: 9 of 21 failures).
        stops = [a for a in order if a.startswith(('send_msg_to_user', 'report_infeasible'))]
        if len(stops) > 1:
            rep = sorted(stops, key=lambda a: (-groups[a]['frequency'], order.index(a)))[0]
            groups[rep] = dict(groups[rep], frequency=sum(groups[a]['frequency'] for a in stops), merged_stops=len(stops))
            order = [a for a in order if a not in stops or a == rep]
    ranked = sorted(order, key=lambda a: (-groups[a]['frequency'], order.index(a)))
    kept = [dict(thought=groups[a]['thought'], action=a, confidence=round(groups[a]['frequency'] / n, 4),
                 frequency=groups[a]['frequency']) for a in ranked[:k]]
    stats = dict(samples=len(replies), parsed=len(replies) - dropped, dropped=dropped, unique=len(order),
                 kept=[(c['action'], c['frequency']) for c in kept], stated_confidence={a: groups[a]['stated_confidence'] for a in ranked[:k]})
    return json.dumps({'BLOCKS': kept}, ensure_ascii=False), stats
