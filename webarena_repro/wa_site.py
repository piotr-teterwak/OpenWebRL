#!/usr/bin/env python3
"""Host one WebArena site without Docker or root: Apptainer sandbox + unprivileged user/net namespace.

    wa_site.py copy  SITE STATE              fresh writable state from the pristine base (per episode)
    wa_site.py serve SITE STATE OUT [--hold SECONDS]
                                             start services in a new loopback-only network namespace,
                                             configure the site origin, write OUT/ready.json, stay up until
                                             OUT/stop or --hold expires
    wa_site.py exec  OUT -- CMD...           run CMD inside the serving namespace (browser, probes)

Recipe from Zixian's validated hosting (scripts/validate_webprm_cms_hosting.py, run_webprm_table5_cms.py):
native in-image fakeroot (no /etc/subuid on this cluster), image Env re-applied after --cleanenv, services on
127.0.0.1 only, Magento base URL rewritten. Differences: the image's own /docker-entrypoint.sh starts
supervisord (its database-init branches are skipped because the /run/*/.init markers exist in the image), and
nginx is moved to the task port so every URL in the task configs works unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

W = Path('/gpfs/scrubbed/piotrt/owrl/webarena')
FAKEROOT = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/native-fakeroot-addon')  # Alpine
FAKEROOT_FOCAL = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/native-fakeroot-addon-focal')
# GitLab overrides (Zixian v9/v9f): no kernel tuning without real root; monitoring off (its reconfigure failed ~25%).
GITLAB_RB = ("\n# --- wa_site.py per-state overrides ---\nexternal_url 'http://localhost:8023'\n"
             "package['modify_kernel_parameters'] = false\nprometheus_monitoring['enable'] = false\n"
             "alertmanager['enable'] = false\n"
             "gitlab_kas['enable'] = false\n"  # Kubernetes agent server: unused by WebArena; failed to start here
             # Peer auth reads the kernel's socket credentials, which fakeroot cannot fake (every process is the
             # namespace root), so map local-socket logins to trust. Custom entries render before the default
             # `local all all peer map=gitlab` line; the socket exists only inside the loopback-only namespace.
             "postgresql['custom_pg_hba_entries'] = { 'wa_namespace_local' => [{ 'type' => 'local', "
             "'database' => 'all', 'user' => 'all', 'cidr' => '', 'method' => 'trust', 'option' => '' }] }\n")
GITLAB_CONTROLLER = r'''umask 022; cd /
/opt/gitlab/embedded/bin/runsvdir -P /opt/gitlab/service "log: ......" &
RSV=$!
sleep 10
if gitlab-ctl reconfigure > /opt/wa-out/reconfigure-0.log 2>&1 || gitlab-ctl reconfigure > /opt/wa-out/reconfigure-1.log 2>&1
then echo ok > /opt/wa-out/reconfigure.status; else echo failed > /opt/wa-out/reconfigure.status; fi
wait $RSV
'''
REDDIT_OCI = Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/webprm-webarena-20261011/local-debug/original-reddit-oci')
SITES = {
    'reddit': dict(base=W / 'sandbox/reddit', oci=REDDIT_OCI, port=9999, origin='http://localhost:9999', probe='/',
                   kind='postmill'),
    'gitlab': dict(base=W / 'sandbox/gitlab', oci=W / 'oci/gitlab', port=8023, origin='http://localhost:8023',
                   probe='/users/sign_in', kind='gitlab', fakeroot=FAKEROOT_FOCAL),
    'cms': dict(base=W / 'sandbox/cms', oci=W / 'oci/cms', port=7780, origin='http://localhost:7780',
                probe='/admin', kind='magento'),
    'shopping': dict(base=W / 'sandbox/shopping', oci=W / 'oci/shopping', port=7770, origin='http://localhost:7770',
                     probe='/', kind='magento'),
}


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def run(argv, timeout=600, **kw):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kw)
    if r.returncode:
        raise RuntimeError(f'{argv[:6]}... rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}')
    return r.stdout


def image_env(site):
    index = json.loads((site['oci'] / 'index.json').read_text())
    manifest = json.loads((site['oci'] / 'blobs/sha256' / index['manifests'][0]['digest'][7:]).read_text())
    config = json.loads((site['oci'] / 'blobs/sha256' / manifest['config']['digest'][7:]).read_text())
    return config['config'].get('Env') or [], config['config'].get('Entrypoint') or [], config['config'].get('Cmd') or []


def prefix(site, state):
    env, _, _ = image_env(site)
    return ['apptainer', 'exec', '--cleanenv', '--no-eval', '--fakeroot', '--ignore-fakeroot-command', '--writable',
            '--containall', '--no-home', '--no-mount', 'bind-paths,home,cwd,hostfs,tmp',
            '--bind', f"{site.get('fakeroot', FAKEROOT)}:/opt/native-fakeroot:ro", str(state), 'env', *env, 'FAKEROOTDONTTRYCHOWN=1',
            '/opt/native-fakeroot/usr/bin/fakeroot', '-l', '/opt/native-fakeroot/usr/lib/libfakeroot.so',
            '-f', '/opt/native-fakeroot/usr/bin/faked', '--']


def cmd_copy(a):
    site = SITES[a.site]; state = Path(a.state)
    # WA_SITE_BASE_<SITE>: a pristine base already unpacked on node-local disk (run_cells.sbatch does this for Shopping,
    # whose 67.6 GB of small files took 2,239 s to copy off GPFS per task, job 410928).
    override = os.environ.get(f"WA_SITE_BASE_{a.site.upper()}")
    if override:
        site = dict(site, base=Path(override))
    if state.exists():
        sys.exit(f'{state} exists')
    t0 = time.time()
    run(['unshare', '--user', '--map-root-user', 'cp', '-a', '--reflink=auto', '--', str(site['base']), str(state)], timeout=3600)
    # Per-state adaptations only; the base stays pristine.
    (state / 'opt/native-fakeroot').mkdir(parents=True, exist_ok=True)
    if site['kind'] == 'postmill':
        # Stale PID/lock files saved in the image would stop PostgreSQL (Zixian's prepare()).
        for name in ('usr/local/pgsql/data/postmaster.pid', 'run/postgresql/.s.PGSQL.5432.lock'):
            p = state / name
            if p.exists():
                p.rename(p.with_name(p.name + '.preserved-before-start'))
        # The authors' configure_websites.sh / refresh_website_dockers.sh remove Postmill's @RateLimit annotations
        # (cache cleared at serve time); WebArena-Verified does the same.
        for name in ('CommentData', 'SubmissionData', 'UserData'):
            run(['sed', '-i', '/@RateLimit/,/)/d', str(state / f'var/www/html/src/DataObject/{name}.php')])
    if site['kind'] in ('magento', 'postmill'):
        conf = state / 'etc/nginx/conf.d/default.conf'
        text = conf.read_text()
        conf.with_name('default.conf.orig').write_text(text)
        conf.write_text(text.replace('listen 80;', f"listen {site['port']};").replace(
            'listen [::]:80 ipv6only=on default_server;', f"listen [::]:{site['port']} ipv6only=on default_server;"))
    if site['kind'] == 'gitlab':
        rb = state / 'etc/gitlab/gitlab.rb'
        text = rb.read_text()
        rb.with_name('gitlab.rb.orig').write_text(text)
        rb.write_text(text.replace("external_url 'http://metis.lti.cs.cmu.edu:8023'", '# (original external_url moved below)') + GITLAB_RB)
        # The custom trust entry above only exists once reconfigure has rendered pg_hba.conf and Postgres reloaded it;
        # when reconfigure's own `CREATE USER` runs first, the pristine `local all all peer` line is still in force and
        # peer auth fails under fakeroot (8 of ~190 GitLab starts in step 1). Put the same trust line in from the start.
        hba = state / 'var/opt/gitlab/postgresql/data/pg_hba.conf'
        hba_text = hba.read_text()
        hba.with_name('pg_hba.conf.orig').write_text(hba_text)
        hba.write_text('local   all         all                               trust  # wa_site.py\n' + hba_text)
    log(f'copied {a.site} -> {state} in {time.time() - t0:.0f} s')


def wait_http(url, deadline):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=20) as r:
                return r.status
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code
            time.sleep(5)  # 502/503 while the app server boots behind nginx
        except Exception:
            time.sleep(3)
    raise TimeoutError(url)


def save_gitlab_logs(state, dest):
    """Last 80 lines of every runit service log (the state copy is deleted after the job)."""
    dest.mkdir(parents=True, exist_ok=True)
    for current in (state / 'var/log/gitlab').glob('*/current'):
        try:
            lines = current.read_text(errors='replace').splitlines()[-80:]
            (dest / (current.parent.name + '.log')).write_text('\n'.join(lines) + '\n')
        except OSError:
            pass


def inside(a):
    """Runs as the mapped root of a fresh user+net namespace."""
    site = SITES[a.site]; state = Path(a.state); out = Path(a.out); P = prefix(site, state)
    run(['ip', 'link', 'set', 'lo', 'up'])
    # Services resolve the node's hostname (Elasticsearch fails without it); the copy may come from another node.
    host = os.uname().nodename
    (state / 'etc/hosts').write_text(f'127.0.0.1 localhost {host} {host.split(".")[0]}\n::1 localhost\n')
    _, entry, cmd = image_env(site)
    # GitLab's runsvdir-start ends in `exec env - PATH=$PATH runsvdir ...`, which wipes LD_PRELOAD/FAKEROOTKEY,
    # so every chpst -U git then fails setgroups. Start runsvdir directly with the environment kept (its ulimit
    # and /proc/sys/fs/file-max lines cannot apply in an unprivileged job anyway). Faked ownership lives in one
    # fakeroot session, so reconfigure must run in the same session as the services it chowns for (otherwise
    # PostgreSQL sees a "wrong ownership" data directory): one container command does both.
    if site['kind'] == 'gitlab':
        P = P[:P.index(str(state))] + ['--bind', f'{out}:/opt/wa-out'] + P[P.index(str(state)):]
        (state / 'opt/wa-out').mkdir(exist_ok=True)
        start = P + ['/bin/sh', '-c', GITLAB_CONTROLLER]
    elif site['kind'] == 'postmill':
        # Faked ownership exists only inside one fakeroot session, so record the owners PostgreSQL checks in the
        # same session that starts it (as Zixian's start_reddit_services.sh), then run the image's own entrypoint.
        start = P + ['/bin/sh', '-c', 'set -e; mkdir -p /run/nginx /run/postgresql /var/tmp/nginx/client_body; '
                     'chown nginx:nginx /run/nginx /var/tmp/nginx /var/tmp/nginx/client_body; '
                     'chown -R postgres:postgres /run/postgresql /usr/local/pgsql/data; exec "$@"', 'start'] + entry + cmd
    else:
        start = P + entry + cmd
    if site['kind'] == 'postmill':  # finish the authors' rate-limit removal (bin/console cache:clear --env=prod)
        run(P + ['sh', '-c', 'cd /var/www/html && php bin/console cache:clear --env=prod'], timeout=900)
    services = subprocess.Popen(start, stdout=open(out / 'services.log', 'w'), stderr=subprocess.STDOUT,
                                start_new_session=True)
    t0 = time.time(); timings = {}
    try:
        if site['kind'] == 'gitlab':
            status = out / 'reconfigure.status'
            while not status.exists():
                if services.poll() is not None or time.time() - t0 > 2400:
                    save_gitlab_logs(state, out / 'svc-logs'); raise RuntimeError('GitLab controller exited or timed out')
                time.sleep(5)
            if status.read_text().strip() != 'ok':
                save_gitlab_logs(state, out / 'svc-logs'); raise RuntimeError('gitlab-ctl reconfigure failed twice; see reconfigure-*.log')
            timings['configured_s'] = round(time.time() - t0, 1)
        if site['kind'] == 'magento':
            env_php = "$a=require '/var/www/magento2/app/etc/env.php'; echo json_encode($a['db']['connection']['default']);"
            db = json.loads(run(P + ['php', '-r', env_php]))
            mysql = P + ['mysql', '-h127.0.0.1', f"-u{db['username']}", f"-p{db['password']}", db['dbname'], '-N', '-e']
            for _ in range(150):
                if services.poll() is not None:
                    raise RuntimeError('services exited during startup; see services.log')
                if subprocess.run(mysql + ['SELECT 1'], capture_output=True, timeout=30).returncode == 0:
                    break
                time.sleep(2)
            else:
                raise TimeoutError('database not ready')
            timings['db_ready_s'] = round(time.time() - t0, 1)
            magento = P + ['php', '/var/www/magento2/bin/magento']
            for argv in (['config:set', 'admin/security/password_is_forced', '0'],
                         ['config:set', 'admin/security/password_lifetime', '0'],
                         ['setup:store-config:set', f"--base-url={site['origin']}/"]):
                run(magento + argv)
            run(mysql + [f"UPDATE core_config_data SET value='{site['origin']}/' WHERE path='web/secure/base_url';"])
            run(magento + ['cache:flush'])
            timings['configured_s'] = round(time.time() - t0, 1)
            wait_http('http://127.0.0.1:9200/_cluster/health?wait_for_status=yellow&timeout=50s', time.time() + 600)
            timings['search_ready_s'] = round(time.time() - t0, 1)
        status = wait_http(site['origin'] + site['probe'], time.time() + 600)
        timings['http_s'] = round(time.time() - t0, 1)
        (out / 'ready.json').write_text(json.dumps(dict(site=a.site, origin=site['origin'], probe_status=status,
                                        pid=os.getpid(), netns=os.readlink('/proc/self/ns/net'), **timings)))
        log('ready', a.site, status, timings)
        deadline = time.time() + a.hold
        while time.time() < deadline and not (out / 'stop').exists() and services.poll() is None:
            time.sleep(1)
        if services.poll() is not None:
            raise RuntimeError('services exited while serving')
    finally:
        try:
            stop = ['gitlab-ctl', 'stop'] if site['kind'] == 'gitlab' else ['supervisorctl', 'shutdown']
            subprocess.run(P + stop, capture_output=True, timeout=240)
        except Exception:
            pass
        try:
            os.killpg(services.pid, signal.SIGTERM); services.wait(timeout=60)
        except Exception:
            try:
                os.killpg(services.pid, signal.SIGKILL)
            except Exception:
                pass
        log('stopped', a.site)


def cmd_serve(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    argv = ['unshare', '--user', '--map-root-user', '--net', '--', sys.executable, __file__, 'inside', a.site, a.state, a.out,
            '--hold', str(a.hold)]
    os.execvp(argv[0], argv)


def cmd_exec(a):
    pid = json.loads((Path(a.out) / 'ready.json').read_text())['pid']
    argv = ['nsenter', f'--target={pid}', '--user', '--net', '--preserve-credentials', '--'] + a.cmd
    os.execvp(argv[0], argv)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('copy'); c.add_argument('site'); c.add_argument('state'); c.set_defaults(f=cmd_copy)
    for name, f in (('serve', cmd_serve), ('inside', inside)):
        s = sub.add_parser(name); s.add_argument('site'); s.add_argument('state'); s.add_argument('out')
        s.add_argument('--hold', type=float, default=3600); s.set_defaults(f=f)
    e = sub.add_parser('exec'); e.add_argument('out'); e.add_argument('cmd', nargs=argparse.REMAINDER); e.set_defaults(f=cmd_exec)
    a = ap.parse_args()
    if getattr(a, 'cmd', None) and a.cmd[:1] == ['--']:
        a.cmd = a.cmd[1:]
    a.f(a)


if __name__ == '__main__':
    main()
