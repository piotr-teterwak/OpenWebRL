#!/usr/bin/env python3
"""Verify paired views and masks; export fixed source-specific validation panels."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

from tokenizers import Tokenizer
from prepare_arm_joint_data import pair_reason, order


# Fields that differ between teacher draws of one state (panel, pick, reasoning, provenance).
DRAW_FIELDS=['id','pair_type','chosen','chosen_index','rejected','rejected_index','draw_id','teacher_reasoning',
             'temperature','candidate_file_offset','candidate_line_sha256','label_sha256','custom_id']


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--cpu-seconds',type=int,default=90)
    a=p.parse_args();start=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2));resource.setrlimit(resource.RLIMIT_CPU,(a.cpu_seconds,a.cpu_seconds+5))
    tok=Tokenizer.from_file('/gpfs/scrubbed/zixianma/checkpoints/web/OpenWebRL-4B-SFT/tokenizer.json')
    counts=Counter();groups={};ids=set();validation=[]
    for split in ['train','validation']:
        groups[split]=set()
        # Extra negatives (prepare --negatives-per-state, ids "<state>#negN", train only) share
        # their state's SFT row; every other pair row aligns one-to-one with the SFT view.
        # Keep digests, not rows: the 384 MiB cap does not fit the whole SFT view in memory.
        def digest_of(x,drop):
            return hashlib.sha256(json.dumps({k:v for k,v in x.items() if k not in drop},sort_keys=True).encode()).digest()
        sft_rows={}
        with (a.data/f'joint_sft.{split}.jsonl').open() as sf:
            for line in sf:
                x=json.loads(line);sft_rows[x['id']]=(digest_of(x,[]),digest_of(x,['id','pair_type']),digest_of(x,DRAW_FIELDS))
        paired=set()
        with (a.data/f'joint_pairs.{split}.jsonl').open() as pf:
            for i,pl in enumerate(pf,1):
                row=json.loads(pl);base,_,extra=row['id'].partition('#')
                assert not extra or split=='train','extra negatives/draws must stay out of validation'
                full,shared,state=sft_rows[base]
                # #drawN rows are another teacher draw of the same state: only state fields match.
                if extra.startswith('draw'):assert digest_of(row,DRAW_FIELDS)==state
                elif extra:assert digest_of(row,['rejected','rejected_index','id','pair_type'])==shared
                else:assert digest_of(row,['rejected','rejected_index'])==full
                if not extra:paired.add(base)
                assert row['id'] not in ids;ids.add(row['id'])
                assert row['split']==split and row['status']=='retained'
                assert pair_reason(row['chosen']['actions'],row['rejected']['actions'])=='eligible'
                assert row['chosen_index']!=row['rejected_index']
                groups[split].add(row['task_group'])
                for name in ['chosen','rejected']:
                    x=row[name];mask=x['action_token_indices']
                    assert mask==sorted(set(mask)) and mask and max(mask)<len(x['token_ids'])
                    assert tok.decode(x['token_ids'],skip_special_tokens=False)==x['text']
                    expected=''.join(x['text'][start:end] for start,end in x['action_char_spans'])+'<|im_end|>'
                    actual=tok.decode([x['token_ids'][k] for k in mask],skip_special_tokens=False)
                    assert actual==expected,(row['id'],name,'mask includes unexpected tokens')
                    assert x['tokens']==len(x['token_ids']) and x['action_tokens']==len(mask)
                    assert row['prompt_tokens']+x['tokens']<=32768
                counts[split+':'+row['source']]+=1
                if split=='validation':validation.append(row)
                if i%100==0:time.sleep(.01)
        assert paired==set(sft_rows),'SFT and pair view length mismatch'
    assert not groups['train']&groups['validation']
    panels={}
    for source in ['C2','Piotr']:
        panel=sorted([x for x in validation if x['source']==source],key=lambda x:order(x['id']+':panel'))[:256]
        path=a.data/f'validation-panel-{source.lower()}.jsonl'
        path.write_text(''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in panel))
        panels[source]={'states':len(panel),'task_groups':len({x['task_group'] for x in panel}),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    download=json.loads((a.data.parent/'joint-data-inputs-0d83b48/download-audit.json').read_text())
    expected={r['path']:r['sha256'] for r in download['files']}
    images=json.loads((a.data/'image-manifest.json').read_text())
    hf_images=0
    for image in images:
        if image['path'] in expected:
            assert image['sha256']==expected[image['path']];hf_images+=1
    usage=resource.getrusage(resource.RUSAGE_SELF)
    report={'passed':True,'counts':dict(counts),'validation_panels':panels,'task_group_overlap':0,
            'hf_image_hashes_matched_download':hf_images,
            'checks':['Exact chosen-response equality between SFT and paired views','Unique state IDs across train/validation',
                      'Disjoint task groups','Eligible action differences','Response token decode equality',
                      'Exact decoded action mask equals tool-call spans plus end-of-turn token','32768 context limit',
                      'HF image bytes match checksum-verified download manifest'],
            'resources':{'wall_seconds':time.monotonic()-start,'cpu_seconds':usage.ru_utime+usage.ru_stime,'peak_rss_mib':usage.ru_maxrss/1024}}
    (a.data/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
