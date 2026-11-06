#!/usr/bin/env python3
"""Functional probes of a hosted WebArena site, run inside its namespace (wa_site.py exec OUT -- python3 probe_site.py SITE).
Prints one JSON line of checks; exit 1 if any check fails. Standard WebArena accounts."""
import http.cookiejar, json, re, sys, urllib.parse, urllib.request

site = sys.argv[1]
jar = http.cookiejar.CookieJar()
op = urllib.request.build_opener(urllib.request.ProxyHandler({}), urllib.request.HTTPCookieProcessor(jar))
checks = {}


def get(url, data=None):
    body = urllib.parse.urlencode(data).encode() if data else None
    with op.open(urllib.request.Request(url, data=body), timeout=120) as r:
        return r.status, r.geturl(), r.read().decode('utf8', 'replace')


if site == 'cms':
    base = 'http://localhost:7780'
    s, u, page = get(base + '/admin')
    key = re.search(r'name="form_key"[^>]*value="([^"]+)"', page)
    checks['login_form'] = bool(key)
    s, u, page = get(base + '/admin/admin/index/index/', dict(form_key=key.group(1) if key else '',
                     **{'login[username]': 'admin', 'login[password]': 'admin1234'}))
    checks['login_dashboard'] = 'Dashboard' in page and 'Sign Out' in page
    s, u, page = get(base + '/admin/sales/order/')
    checks['orders_grid'] = s == 200 and 'Orders' in page
    s, u, page = get(base + '/catalogsearch/result/?q=shirt')
    checks['storefront_search'] = s == 200 and ('product-item' in page or 'search results' in page.lower())
if site == 'shopping':
    base = 'http://localhost:7770'
    s, u, page = get(base + '/customer/account/login/')
    key = re.search(r'name="form_key"[^>]*value="([^"]+)"', page)
    checks['login_form'] = bool(key)
    s, u, page = get(base + '/customer/account/loginPost/', {'form_key': key.group(1) if key else '',
                     'login[username]': 'emma.lopez@gmail.com', 'login[password]': 'Password.123'})
    s, u, page = get(base + '/customer/account/')
    checks['login_ok'] = 'My Account' in page and 'Sign Out' in page
    s, u, page = get(base + '/sales/order/history/')
    checks['order_history'] = s == 200 and 'order' in page.lower()
    s, u, page = get(base + '/catalogsearch/result/?q=headphones')
    checks['storefront_search'] = s == 200 and 'product-item' in page
if site == 'reddit':
    base = 'http://localhost:9999'
    s, u, page = get(base + '/login')
    token = re.search(r'name="_csrf_token" value="([^"]+)"', page)
    checks['login_form'] = bool(token)
    s, u, page = get(base + '/login_check', {'_csrf_token': token.group(1) if token else '',
                     '_username': 'MarvelsGrantMan136', '_password': 'test1234'})
    checks['login_ok'] = 'MarvelsGrantMan136' in page and 'Log out' in page
    s, u, page = get(base + '/forums')
    checks['forums_listed'] = s == 200 and page.count('/f/') > 20
    s, u, page = get(base + '/f/books')
    checks['books_forum'] = s == 200 and 'submission' in page.lower()
if site == 'gitlab':
    base = 'http://localhost:8023'
    s, u, page = get(base + '/users/sign_in')
    token = re.search(r'name="authenticity_token" value="([^"]+)"', page)
    checks['login_form'] = bool(token)
    s, u, page = get(base + '/users/sign_in', {'authenticity_token': token.group(1) if token else '',
                     'user[login]': 'byteblaze', 'user[password]': 'hello1234', 'user[remember_me]': '0'})
    checks['login_ok'] = 'Sign out' in page or 'sign_out' in page
    with op.open(base + '/api/v4/projects?per_page=1&statistics=false', timeout=120) as r:
        total = r.headers.get('X-Total')
    checks['projects_listed'] = bool(total) and int(total) > 0
    s, u, page = get(base + '/a11yproject/a11yproject.com/-/issues')
    checks['a11y_issues_page'] = s == 200 and 'Issues' in page
    extra = dict(public_projects=total)
else:
    extra = {}
print(json.dumps(dict(site=site, **checks, **extra)))
sys.exit(0 if checks and all(checks.values()) else 1)
