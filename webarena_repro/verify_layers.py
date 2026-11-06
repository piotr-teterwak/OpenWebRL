#!/usr/bin/env python3
"""Verify an OCI image layout end to end against an expected config digest.

    verify_layers.py OCI_DIR EXPECTED_CONFIG_DIGEST REPORT.json

Checks: the manifest's config digest equals EXPECTED (the digest read from the original WebArena tar, see
verify/tar_index.py); the config blob hashes to it; every compressed layer blob hashes to its manifest digest
and size; and every layer, decompressed, hashes to the config's rootfs.diff_ids entry at the same position.
Decompression uses `gzip -dc` / `zstd -dc` streamed into sha256 (no files written). Exit 0 only if all pass.
"""
import hashlib, json, os, subprocess, sys, time

oci, expected, report = sys.argv[1], sys.argv[2], sys.argv[3]
blobs = os.path.join(oci, 'blobs', 'sha256')
index = json.load(open(os.path.join(oci, 'index.json')))
mdig = index['manifests'][0]['digest'][7:]
man = json.load(open(os.path.join(blobs, mdig)))
cdig = man['config']['digest']
cfg_raw = open(os.path.join(blobs, cdig[7:]), 'rb').read()
diff_ids = json.loads(cfg_raw)['rootfs']['diff_ids']
res = dict(oci=oci, manifest=mdig, config_digest=cdig, expected_config_digest=expected,
           config_matches_original=(cdig == expected), config_blob_ok=hashlib.sha256(cfg_raw).hexdigest() == cdig[7:],
           layer_count_matches=len(man['layers']) == len(diff_ids), layers=[])


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
    return h.hexdigest()


def sha_decompressed(path, media):
    tool = ['zstd', '-dc'] if 'zstd' in media else ['gzip', '-dc'] if ('gzip' in media or media.endswith('.tar.gzip')) else ['cat']
    p = subprocess.Popen(tool + [path], stdout=subprocess.PIPE)
    h = hashlib.sha256(); n = 0
    for b in iter(lambda: p.stdout.read(1 << 24), b''): h.update(b); n += len(b)
    if p.wait(): raise RuntimeError(f'{tool[0]} failed on {path}')
    return h.hexdigest(), n


t0 = time.time()
for i, layer in enumerate(man['layers']):
    path = os.path.join(blobs, layer['digest'][7:])
    row = dict(index=i, digest=layer['digest'], media=layer.get('mediaType'), size=layer['size'],
               size_ok=os.path.getsize(path) == layer['size'])
    row['compressed_ok'] = sha_file(path) == layer['digest'][7:]
    got, n = sha_decompressed(path, layer.get('mediaType', ''))
    row.update(diff_id='sha256:' + got, expected_diff_id=diff_ids[i] if i < len(diff_ids) else None,
               uncompressed_bytes=n)
    row['diff_id_ok'] = row['diff_id'] == row['expected_diff_id']
    res['layers'].append(row)
    print(f"[{i + 1}/{len(man['layers'])}] {layer['digest'][7:19]} compressed={row['compressed_ok']} "
          f"diff_id={row['diff_id_ok']} {n / 1e9:.1f} GB ({time.time() - t0:.0f} s)", flush=True)
res['all_ok'] = (res['config_matches_original'] and res['config_blob_ok'] and res['layer_count_matches']
                 and all(r['size_ok'] and r['compressed_ok'] and r['diff_id_ok'] for r in res['layers']))
res['seconds'] = round(time.time() - t0)
json.dump(res, open(report, 'w'), indent=1)
print('ALL_OK' if res['all_ok'] else 'VERIFY_FAILED', oci)
sys.exit(0 if res['all_ok'] else 1)
