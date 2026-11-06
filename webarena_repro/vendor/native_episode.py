# Vendored from Zixian Ma's native-browser-v3/native_episode.py (2026-10-26); only paths/imports changed.
"""Executable prospective native CoordinatorAgent/ScriptBrowserEnv integration.

No API client, endpoint, allocation or budget is created here. The owner injects
an approved completion_scope with assert_healthy(), a resolved task config, and
an evaluator callback. Entire upstream coordinator, post-action mixin, browser
environment, action module, processors and history formatter are imported intact.
Only unused AgentScope constructors, persistence and browser launch plumbing are
adapted. This is not a recovered historical benchmark runner.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import importlib
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from urllib.parse import urlsplit

from webprm_native_release import load_tools, namespace, reference  # vendored from Zixian's openwebrl/ (arm @ 763753a)

SOURCE = Path('/gpfs/scrubbed/piotrt/owrl/webarena/webarbiter-764ec26')  # YaoZ720/WebArbiterCode @ 764ec26 (identical to Zixian's copy)
DEPENDENCIES = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/next-stage-preparation/webarena-subset-review-20261021/runtime-dependencies-v1/site-packages')
CHROMIUM = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/browsers/chromium-1208/chrome-linux64/chrome')
_episode_lock = threading.Lock()


def writable_root(path):
    root=Path(path).resolve()
    allowed=[Path('/tmp'),Path('/gpfs/scrubbed/piotrt/owrl/webarena')]
    if not any(root.is_relative_to(parent) for parent in allowed):
        raise ValueError('Native evidence path is outside approved writable roots')
    return root


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(x) for x in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def write_json(path, value):
    path = Path(path)
    with path.open('x') as handle:
        json.dump(plain(value), handle, sort_keys=True, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    return reference(path)


def stage_chromium(source_executable=CHROMIUM, *, cache_root='/tmp/webprm-native-chromium-20261021'):
    """Byte-verify a node-local existing browser tree; zero network downloads.

    Keeps the original tree intact. Every copied byte remains represented in a
    receipt; interrupted/changed caches fail closed rather than being erased.
    """
    source=Path(source_executable).resolve().parent
    files=[];total=0
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Browser cache symlink requires separate review')
        if path.is_file():
            row=reference(path);size=path.stat().st_size;total+=size
            files.append(dict(relative=str(path.relative_to(source)),sha256=row['sha256'],bytes=size))
    if not files or total>1024**3:
        raise RuntimeError('Existing browser staging exceeds1GiB envelope')
    manifest=dict(source=str(source),files=files,total_bytes=total)
    identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cache=writable_root(cache_root);cache.mkdir(parents=True,exist_ok=True)
    destination=cache/identity[:24]
    receipt=destination/'staging-receipt.json'
    copied=False
    if not destination.exists():
        destination.mkdir();shutil.copytree(source,destination/'chrome-linux64',symlinks=False)
        copied=True
    target=destination/'chrome-linux64'
    actual={str(p.relative_to(target)) for p in target.rglob('*') if p.is_file()}
    if actual!={r['relative'] for r in files}:
        raise RuntimeError('Local browser tree file inventory differs')
    for row in files:
        path=target/row['relative']
        if path.is_symlink() or path.stat().st_size!=row['bytes'] or reference(path)['sha256']!=row['sha256']:
            raise RuntimeError('Local browser bytes differ from existing pinned source')
    if copied:
        write_json(receipt,dict(verified=True,identity_sha256=identity,manifest=manifest,
            local_copy_bytes=total,new_network_bytes=0,gpu_seconds=0,new_allocations=0))
    else:
        saved=json.loads(receipt.read_text())
        if saved.get('identity_sha256')!=identity or saved.get('manifest')!=manifest:
            raise RuntimeError('Local browser staging receipt changed')
    return dict(executable=str(target/Path(source_executable).name),receipt=reference(receipt),
                identity_sha256=identity,local_copy_bytes_this_attempt=total if copied else 0,
                retained_local_bytes=total,new_network_bytes=0)


class UnusedAgentScopeConstructor:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Unused AgentScope/UI constructor must not execute')


def _unused_module(name, fields):
    if name not in sys.modules:
        module = ModuleType(name)
        for field in fields:
            setattr(module, field, UnusedAgentScopeConstructor)
        sys.modules[name] = module


def load_native(source=SOURCE):
    """Whole-module loading; no extracted/rewritten coordinator/browser methods."""
    source = Path(source).resolve()
    if str(DEPENDENCIES) not in sys.path:
        sys.path.insert(0, str(DEPENDENCIES))
    # URL-mapping module imports httpx even though this runner never asks it to
    # perform network discovery. Reuse genuine already-installed pure Python
    # dependencies rather than replacing modules or downloading a new backend.
    existing_site=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/venv/lib/python3.12/site-packages')
    if str(existing_site) not in sys.path:
        sys.path.append(str(existing_site))
    os.environ.setdefault('DATASET', 'webarena')
    if os.environ['DATASET'] != 'webarena':
        raise RuntimeError('Explicit WebArena dataset required')
    tools, refs = load_tools(source)
    namespace('browser_env', source/'viewer/src/browser_env')
    # Alias the same real modules: native evaluator imports this second prefix.
    browser = importlib.import_module('browser_env.envs')
    actions = importlib.import_module('browser_env.actions')
    for suffix in ('actions', 'constants', 'processors', 'utils', 'envs'):
        sys.modules['webarbiter.src.browser_env.'+suffix] = importlib.import_module('browser_env.'+suffix)
    namespace('webarbiter.src.browser_env', source/'viewer/src/browser_env')
    namespace('webarbiter.src.server.agents', source/'viewer/src/server/agents')
    # execute_step never invokes these objects. Avoid constructing independent
    # OpenAIChatModel clients or importing missing UI-only AgentScope services.
    _unused_module('agentscope.pipeline', ['MsgHub'])
    _unused_module('webarbiter.src.server.core.memory', ['RunMemory'])
    for suffix, cls in [('browser_agent','BrowserAgent'), ('reranking_agent','RerankingAgent'),
                        ('sampling_agent','SamplingAgent'), ('tournament_agent','TournamentAgent')]:
        _unused_module('webarbiter.src.server.agents.'+suffix, [cls])
    coordinator = importlib.import_module('webarbiter.src.server.agents.coordinator_agent')
    post = importlib.import_module('webarbiter.src.server.service.step_management')
    namespace('webarbiter.src.server.service.executors',source/'viewer/src/server/service/executors')
    executor=importlib.import_module('webarbiter.src.server.service.executors.step_loop_executor')
    terminal_handler=importlib.import_module('webarbiter.src.server.service.utils.terminal_action_handler')
    evaluation_trigger=importlib.import_module('webarbiter.src.server.service.utils.evaluation_trigger')
    history = tools['history']
    for module in (coordinator, post, browser, actions, history, executor, terminal_handler):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError('Native module escaped pinned source')
    return SimpleNamespace(tools=tools, coordinator=coordinator, post=post,
                           executor=executor,terminal_handler=terminal_handler,evaluation_trigger=evaluation_trigger,
                           history=history, browser=browser, actions=actions,
                           refs=refs + [reference(m.__file__) for m in (coordinator, post, browser, actions, history,
                               executor,terminal_handler,evaluation_trigger,
                               importlib.import_module('browser_env.processors'),importlib.import_module('browser_env.utils'))])


class ReceiptRepository:
    """Persistence boundary consumed by unchanged native history/post-action code."""
    def __init__(self, run_id):
        self.run_id, self.steps, self.states = run_id, [], []

    def get_steps(self, *, run_id, start, limit):
        if run_id != self.run_id:
            raise RuntimeError('Run identity mismatch')
        return deepcopy(self.steps[start:start+limit])

    def get_trajectory_tree(self, run_id):
        if run_id != self.run_id:
            raise RuntimeError('Run identity mismatch')
        return SimpleNamespace(get_state_nodes=lambda: self.states)

    def update_run_status(self,**kwargs):
        self.run_status=deepcopy(kwargs)

    def get_run_info(self,run_id):
        if run_id!=self.run_id:raise RuntimeError('Run identity mismatch')
        return dict(task_id='owned-native-episode',run_id=run_id)


class ReceiptMemory:
    def __init__(self):
        self.steps = []

    async def add_step(self, **kwargs):
        self.steps.append(deepcopy(kwargs))


class ReceiptEvents:
    def __init__(self, root, repository):
        self.root, self.repository, self.rows = root, repository, []

    async def emit(self, event):
        row = plain(event.to_dict())
        self.rows.append(row)
        write_json(self.root/f'event-{len(self.rows):04d}.json', row)
        if row.get('type') == 'post_action_complete':
            node = next(n for n in self.repository.states if n.step == row['step'])
            action = row.get('terminal_action')
            # Same persistence condition as native post_action_handlers.py:
            # incomplete action=None blocks do not become terminal candidates.
            node.terminal_action_candidate = action if action and action.get('action') is not None else None


class _LaunchProxy:
    def __init__(self, original, root, executable=CHROMIUM):
        self.original, self.root, self.executable = original, root, executable

    def launch(self, **kwargs):
        if 'executable_path' in kwargs:
            raise RuntimeError('Unexpected upstream executable override')
        kwargs['executable_path'] = str(self.executable)
        kwargs['env'] = dict(os.environ, TMPDIR=str(self.root), XDG_CACHE_HOME=str(self.root/'cache'))
        return self.original.launch(**kwargs)


class _PlaywrightContext:
    def __init__(self, original, root):
        self.original, self.root = original, root

    def driver_enter(self):
        # The Node driver creates the Chromium profile before Chromium's own
        # launch env is applied. Set its inherited TMPDIR at driver startup.
        previous=os.environ.get('TMPDIR')
        os.environ['TMPDIR']=str(self.root)
        try:return self.original.__enter__()
        finally:
            if previous is None:os.environ.pop('TMPDIR',None)
            else:os.environ['TMPDIR']=previous

    def __enter__(self):
        pw = self.driver_enter()
        return SimpleNamespace(chromium=_LaunchProxy(pw.chromium, self.root))

    def __exit__(self, *args):
        return self.original.__exit__(*args)


# v2 (2026-10-23): the released WebArbiter service runs ScriptBrowserEnv with 1.5 s
# (viewer/src/server/service/run_management.py:177). v1 used 0, so a navigating click
# raced _get_obs's page.evaluate ('Execution context was destroyed'; job 383391).
SLEEP_AFTER_EXECUTION = 1.5
GEOMETRY_ATTEMPTS = 20


class NativeBrowser:
    """Original synchronous ScriptBrowserEnv on one owned affinity thread.

    A short /tmp launch directory avoids Chromium UNIX socket path limits. The
    original environment and action/processor bodies remain unchanged. Geometry
    is recorded without changing it by default. A separately declared opt-in
    compatibility mode can align CDP screen/viewport before the native filter.
    """
    def __init__(self, native, root, *, viewport, allowed_origins=(), local_fixture=False,
                 compatibility_geometry=False, chromium=CHROMIUM, launch_parent=None):
        self.native, self.root = native, writable_root(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.viewport, self.allowed_origins = dict(viewport), frozenset(allowed_origins)
        self.local_fixture = local_fixture
        self.compatibility_geometry = compatibility_geometry
        self.chromium=Path(chromium).resolve()
        parent=Path(launch_parent or os.environ.get('WEBPRM_BROWSER_TMP_ROOT','/tmp'))
        if not parent.is_absolute() or parent.resolve()!=parent or not parent.is_relative_to(Path('/tmp')) or not parent.is_dir():
            raise ValueError('Browser launch parent must be an existing canonical non-symlink directory beneath /tmp')
        self.launch_parent=parent
        self.profile_directories=[]
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='native-browser')
        self.env, self.observations, self.trajectory = None, [], []
        self.closed, self.launch_root = False, None
        self.geometry_retries = 0
        self.lazy_page_clients = 0  # v3: CDP sessions created for tabs opened by actions

    async def call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def _route(self, route):
        split = urlsplit(route.request.url)
        origin = f'{split.scheme}://{split.netloc}'
        if not self.local_fixture and origin in self.allowed_origins:
            route.continue_()
        else:
            route.abort('blockedbyclient')

    def _start(self, config_file):
        import tempfile
        from playwright.sync_api import sync_playwright
        self.launch_root = Path(tempfile.mkdtemp(prefix='wab-', dir=str(self.launch_parent)))
        env_module, native = self.native.browser, self.native
        viewport, route_callback = self.viewport, self._route
        class AlignedEnvironment(env_module.ScriptBrowserEnv):
            def setup(env, config_file=None):
                # Native setup must route before navigation, so supply a context
                # proxy whose new_context installs the allowlist before new_page.
                return super().setup(config_file=config_file)

            def get_page_client(env, page):
                # v3 (2026-10-26): the released setup creates `page.client` (a CDP session with the
                # device-metrics override and Accessibility.enable) only for pages opened at reset, so an
                # action that opens a new tab ended the episode with AttributeError 'Page' object has no
                # attribute 'client' (v8: 4/54 cells). Create the same session lazily, once per page.
                client = getattr(page, 'client', None)
                if client is None:
                    client = page.context.new_cdp_session(page)
                    client.send('Emulation.setDeviceMetricsOverride', dict(width=env.viewport_size['width'],
                        height=env.viewport_size['height'], deviceScaleFactor=1, mobile=False))
                    if env.text_observation_type == 'accessibility_tree':
                        client.send('Accessibility.enable')
                    page.client = client
                    self.lazy_page_clients += 1
                return client

            def _get_obs(env):
                if env.page.viewport_size != viewport:
                    raise RuntimeError('Native viewport identity changed')
                if self.compatibility_geometry:
                    client = env.get_page_client(env.page)
                    client.send('Emulation.clearDeviceMetricsOverride')
                    client.send('Emulation.setDeviceMetricsOverride', dict(width=viewport['width'],
                        height=viewport['height'],screenWidth=viewport['width'],screenHeight=viewport['height'],
                        deviceScaleFactor=1,mobile=False))
                # v2: geometry recording is our addition, not the original observation. A
                # navigation committing under it destroys the JS context; wait for the
                # document and retry only that transient error (bounded).
                for attempt in range(GEOMETRY_ATTEMPTS):
                    try:
                        env.page.wait_for_load_state('domcontentloaded', timeout=15000)
                        self.geometry = env.page.evaluate('({screenWidth:screen.width,screenHeight:screen.height,'
                            'innerWidth:innerWidth,innerHeight:innerHeight,devicePixelRatio:devicePixelRatio})')
                        break
                    except Exception as error:
                        if 'Execution context was destroyed' not in str(error) or attempt == GEOMETRY_ATTEMPTS-1:
                            raise
                        self.geometry_retries += 1
                        time.sleep(.25)
                if self.compatibility_geometry and self.geometry != dict(screenWidth=viewport['width'],screenHeight=viewport['height'],
                    innerWidth=viewport['width'],innerHeight=viewport['height'],devicePixelRatio=1):
                    raise RuntimeError('Native geometry alignment failed')
                return super()._get_obs()
        self.env = AlignedEnvironment(headless=True, observation_type='accessibility_tree',
            current_viewport_only=True, viewport_size=viewport, sleep_after_execution=SLEEP_AFTER_EXECUTION)
        original_sync = env_module.sync_playwright
        # Whole native setup is called. Supply launch and new_context plumbing,
        # rather than duplicating setup, observations or action execution.
        owner = self
        class BrowserProxy:
            def __init__(self, browser): self.browser=browser
            def new_context(self, **kwargs):
                context=self.browser.new_context(**kwargs)
                context.route('**/*', route_callback)
                return context
            def __getattr__(self,name): return getattr(self.browser,name)
        class LaunchProxy(_LaunchProxy):
            def __init__(self,original,root):super().__init__(original,root,owner.chromium)
            def launch(self, **kwargs): return BrowserProxy(super().launch(**kwargs))
        class ContextProxy(_PlaywrightContext):
            def __enter__(self):
                pw=self.driver_enter()
                return SimpleNamespace(chromium=LaunchProxy(pw.chromium,self.root))
        env_module.sync_playwright=lambda:ContextProxy(sync_playwright(),owner.launch_root)
        try:
            observation, info=self.env.reset(options=dict(config_file=str(config_file)))
            self.last=(observation,info)
            self.trajectory=[deepcopy(dict(observation=observation, info=info))]
            self.profile_directories=[str(p) for p in self.launch_root.glob('playwright_chromiumdev_profile-*') if p.is_dir()]
            if len(self.profile_directories)!=1:
                raise RuntimeError('Chromium profile did not stay inside the owned launch directory')
            return self._capture(observation,info)
        finally:
            env_module.sync_playwright=original_sync

    def _capture(self, observation, info):
        from PIL import Image
        index=len(self.observations)
        image=self.root/f'observation-{index:03d}.png'
        Image.fromarray(observation['image']).save(image)
        nodes=deepcopy(info['observation_metadata']['text']['obs_nodes_info'])
        row=dict(index=index,text=observation['text'],obs_nodes_info=nodes,obs_node_order=list(nodes),url=self.env.page.url,
                 fail_error=info.get('fail_error',''),geometry=deepcopy(self.geometry),image=reference(image),
                 image_pixels=dict(sha256=hashlib.sha256(observation['image'].tobytes()).hexdigest(),
                    shape=list(observation['image'].shape),dtype=str(observation['image'].dtype)))
        row['receipt']=write_json(self.root/f'observation-{index:03d}.json',row)
        self.observations.append(row)
        return deepcopy(row)

    async def start(self, config_file):
        return await self.call(self._start, config_file)

    def _execute(self, action_string):
        try:
            action=self.native.actions.create_id_based_action(action_string)
        except self.native.actions.ActionParsingError:
            # (repro adaptation) WebArena's run.py executes a none action on an unparseable action and the episode
            # continues; the wrapper used to raise, which voided the cell. Parse failures are agent behavior.
            self.parse_failures=getattr(self,'parse_failures',0)+1
            action=self.native.actions.create_none_action()
            action['raw_prediction']=action_string
        if action['action_type']==self.native.actions.ActionTypes.SEND_MESSAGE_TO_USER:
            # Match the original service terminal boundary. Its browser action
            # executor deliberately does not handle SEND_MESSAGE_TO_USER.
            # Preserve the selected native action, but never call env.step or
            # create another observation/state for a message to the user.
            self.trajectory.append(deepcopy(action))
            observed=deepcopy(self.observations[-1])
            dispatch=write_json(self.root/f'terminal-dispatch-{len(self.trajectory):03d}.json',dict(
                source=reference(self.native.post.__file__),
                method='StepManagementMixin.execute_action_and_append_state',
                rule='Faithful terminal boundary adapter: parsed SEND_MESSAGE_TO_USER skips browser and new state',
                whole_dispatch_method_invoked=False,no_browser_step=True,no_new_state=True,
                action_string=action_string,raw_action=plain(action),observation=observed['receipt']))
            return dict(action=plain(action),observation=observed,terminal=True,
                        answer=action.get('answer'),terminal_dispatch=dispatch)
        observation,_,_,_,info=self.env.step(action)
        # Native processor metadata is a mutable singleton. Freeze the actual
        # action-time state here, before a later observation can mutate it.
        self.trajectory.extend([deepcopy(action),deepcopy(dict(observation=observation,info=info))])
        row=self._capture(observation,info)
        return dict(action=plain(action),observation=row,
                    terminal=action['action_type']==self.native.actions.ActionTypes.SEND_MESSAGE_TO_USER,
                    answer=action.get('answer') if action['action_type']==self.native.actions.ActionTypes.SEND_MESSAGE_TO_USER else None)

    async def execute(self, action_string):
        return await self.call(self._execute, action_string)

    async def evaluate(self, callback, task, terminal):
        return await self.call(callback, task, self.trajectory, self.env.page, terminal)

    def _close(self):
        errors=[]
        if self.env is not None:
            for attr in ('context','browser'):
                value=getattr(self.env,attr,None)
                if value is not None:
                    try:value.close()
                    except Exception as error:errors.append(dict(stage=attr,error_type=type(error).__name__))
            try:
                if hasattr(self.env,'context_manager'):
                    self.env.context_manager.__exit__()
            except Exception as error:errors.append(dict(stage='playwright',error_type=type(error).__name__))
        self.closed=True
        return dict(closed=True,errors=errors,launch_root=str(self.launch_root),
                    observations=len(self.observations),browser_class='Original ScriptBrowserEnv subclass: geometry recording',
                    compatibility_geometry=self.compatibility_geometry,chromium=str(self.chromium),
                    launch_parent=str(self.launch_parent),profile_directories=self.profile_directories)

    async def close(self):
        try:return await self.call(self._close)
        finally:self.executor.shutdown(wait=True,cancel_futures=False)


async def run_episode(*, task, context, browser, completion_scope, evaluate,
                      evidence_root, max_steps=30, native=None, decision_adapter=None):
    """Run one bounded episode; caller must own reset, financial caps and evaluation.

    evaluate(task, native_trajectory, native_page, terminal_record) runs on the
    browser thread before close. It must retain the original native/effective
    verdict separately; this function does not invent a missing terminal answer.
    """
    if isinstance(max_steps,bool) or not isinstance(max_steps,int) or not 1<=max_steps<=30:
        raise ValueError('Bounded1..30 native steps required')
    if task.get('intent')!=context.intent or not callable(evaluate):
        raise ValueError('Exact task intent and explicit evaluator required')
    root=writable_root(evidence_root);root.mkdir(parents=True,exist_ok=False)
    if not _episode_lock.acquire(blocking=False):
        raise RuntimeError('One native episode per process required')
    native=native or browser.native
    repo=ReceiptRepository(context.run_id);hub=ReceiptEvents(root,repo)
    agent=native.coordinator.CoordinatorAgent.__new__(native.coordinator.CoordinatorAgent)
    agent.run_context=context;agent.memory=ReceiptMemory()
    post=native.post.StepManagementMixin.__new__(native.post.StepManagementMixin)
    post._trajectory_repository=repo
    post.get_step_info=lambda run_id,step: repo.states[step] if 0<=step<len(repo.states) else None
    post._get_trajectory_document=lambda run_id:context
    original_hubs=(native.coordinator.event_hub,native.post.event_hub,
                   native.executor.event_hub,native.evaluation_trigger.event_hub)
    native.coordinator.event_hub=native.post.event_hub=native.executor.event_hub=native.evaluation_trigger.event_hub=hub
    result=dict(status='failed',valid=False,task_id=task.get('task_id'),run_id=context.run_id,
                selected_actions=[],terminal=None,verdict=None)
    async def execute_selected(action_string,step):
        execution=await browser.execute(action_string)
        if execution['terminal']:
            # The original service spawns unowned event threads. Await its
            # unchanged terminal event method under this owned episode instead.
            await post._emit_terminal_post_action_complete(context.run_id,step,action_string,
                dict(action=action_string,thought='',confidence=1.,status='selected'))
        return execution
    try:
        config=root/'resolved-task.json';write_json(config,task)
        observation=await browser.start(config)
        repo.states.append(SimpleNamespace(step=0,url=observation['url'],
            obs_nodes_info=observation['obs_nodes_info'],terminal_action_candidate=None))
        state_diff='';previous_terminal=None
        with completion_scope as bridge:
            if not callable(getattr(bridge,'assert_healthy',None)):
                raise RuntimeError('Metered completion scope with health latch required')
            for step in range(max_steps):
                bridge.assert_healthy()
                history=native.history.get_trajectory_text(repo,context.run_id,step)
                inputs=dict(current_step=step,observation_text=observation['text'],
                    current_url=observation['url'],trajectory_text=history,state_diff=state_diff,
                    previous_step_terminal_action=previous_terminal)
                decision=await (agent.execute_step(**inputs) if decision_adapter is None
                    else decision_adapter(native=native,agent=agent,context=context,inputs=inputs))
                bridge.assert_healthy() # native helpers may swallow transport failures
                write_json(root/f'decision-{step:03d}.json',dict(history=history,
                    input_observation=observation['receipt'],state_diff=state_diff,decision=plain(decision)))
                selected=plain(decision.get('selected_action'))
                if not selected or not selected.get('action'):
                    selected=native.terminal_handler.TerminalActionHandler.extract_action(
                        decision,fallback_action=previous_terminal)
                    write_json(root/f'no-selection-fallback-{step:03d}.json',dict(selected=selected,
                        source=reference(native.terminal_handler.__file__),
                        rule='Whole native TerminalActionHandler.extract_action'))
                execution=await execute_selected(selected['action'],step)
                result['selected_actions'].append(dict(step=step,selected=selected,execution=execution))
                repo.steps.append(dict(step=step,selected_action=selected,obs_nodes_info=observation['obs_nodes_info']))
                observation=execution['observation']
                if execution['terminal']:
                    result['terminal']=dict(answer=execution['answer'],action=selected['action'],step=step)
                    result['status']='terminal'
                    break
                repo.states.append(SimpleNamespace(step=step+1,url=observation['url'],
                    obs_nodes_info=observation['obs_nodes_info'],terminal_action_candidate=None))
                before=len(hub.rows)
                await post._compute_post_action_async(context.run_id,step+1,selected['action'],
                    SimpleNamespace(url=observation['url'],obs_nodes_info=observation['obs_nodes_info']))
                bridge.assert_healthy()
                complete=[r for r in hub.rows[before:] if r.get('type')=='post_action_complete' and r.get('step')==step+1]
                if len(complete)!=1:
                    raise RuntimeError('Native post-action stage did not complete')
                state_diff=complete[0].get('diff_text') or ''
                previous_terminal=repo.states[-1].terminal_action_candidate
            else:
                result['status']='turn_limit'
            bridge.assert_healthy()
            if result['terminal'] is None:
                # Execute the ORIGINAL fallback method. Persistence/browser
                # service callbacks are adapters; no STOP is invented by the
                # evaluator, and the extra action uses native parser/env.step.
                loop=asyncio.get_running_loop();executions=[]
                def execute_fallback(*,run_id,action,current_state):
                    value=asyncio.run_coroutine_threadsafe(execute_selected(action,current_state.step),loop).result()
                    executions.append(value)
                fallback=native.executor.StepLoopExecutor.__new__(native.executor.StepLoopExecutor)
                fallback.service=SimpleNamespace(get_step_info=lambda run_id,step:repo.states[step],
                    execute_action_and_append_state=execute_fallback,_trajectory_repository=repo)
                began=time.time()
                await fallback._handle_na_fallback(context.run_id,max_steps,max_steps-1,began,0.)
                bridge.assert_healthy()
                if len(executions)!=1 or not executions[0]['terminal'] or executions[0]['answer']!='N/A':
                    raise RuntimeError('Original native N/A fallback did not execute and close')
                execution=executions[0]
                result['selected_actions'].append(dict(step=max_steps,
                    selected=dict(action='send_msg_to_user [N/A]',thought='Native max-step fallback'),
                    execution=execution,fallback=True))
                result['terminal']=dict(answer=execution['answer'],action='send_msg_to_user [N/A]',
                    step=max_steps,fallback=True,source=reference(native.executor.__file__),
                    rule='Whole StepLoopExecutor._handle_na_fallback')
                result['status']='max_steps_reached_na_fallback'
                write_json(root/'native-na-fallback.json',dict(terminal=result['terminal'],execution=execution,
                    source=reference(native.executor.__file__),repository_status=repo.run_status))
            result['verdict']=plain(await browser.evaluate(evaluate,task,result['terminal']))
            bridge.assert_healthy()
            result['valid']=True
    except Exception as error:
        result.update(status='failed',valid=False,error_type=type(error).__name__,error=str(error))
    finally:
        try:result['closure']=await browser.close()
        except Exception as error:
            result.update(valid=False,status='closure_failed',closure_error_type=type(error).__name__)
        (native.coordinator.event_hub,native.post.event_hub,native.executor.event_hub,
            native.evaluation_trigger.event_hub)=original_hubs
        _episode_lock.release()
        result['native_sources']=native.refs
        # Lossless screenshot files and raw pixel digests retain the exact
        # native image arrays without enormous repeated JSON channel lists.
        # Actual native action dictionaries and state ordering are unchanged.
        raw=[];state_index=0
        for item in browser.trajectory:
            if isinstance(item,dict) and 'observation' in item:
                if state_index>=len(browser.observations):
                    result.update(valid=False,status='incomplete_native_capture')
                    raw.append(dict(incomplete_capture=True,observation_text=item['observation'].get('text'),
                        image_pixels_sha256=hashlib.sha256(item['observation']['image'].tobytes()).hexdigest()))
                    state_index+=1
                    continue
                observed=browser.observations[state_index]
                raw.append(dict(observation=dict(text=item['observation']['text'],
                    image=observed['image'],image_pixels=observed['image_pixels']),
                    info=plain(item['info']),state_observation_receipt=observed['receipt']))
                state_index+=1
            else:raw.append(plain(item))
        result['raw_native_trajectory']=write_json(root/'raw-native-trajectory.json',raw)
        result['runtime_adaptations']=['Bypass unused AgentScope constructors; receipt memory/persistence replaces UI/DB.',
            'Whole CoordinatorAgent.execute_step and StepManagementMixin post-action methods unchanged.',
            'Whole ScriptBrowserEnv/actions/processors; explicit cached Chromium; geometry is recorded without override by default.',
            'Original service terminal boundary faithfully adapted: actual parsed SEND_MESSAGE_TO_USER appends action only; no env.step/new state; original terminal event method awaited instead of daemon threads.',
            'One affinity browser thread and fully awaited model/helper work; external task reset/evaluator/budget callback required.']
        write_json(root/'result.json',result)
    return result
