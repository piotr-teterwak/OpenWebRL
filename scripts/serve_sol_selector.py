#!/usr/bin/env python3
"""Bounded GPT-5.6 Sol selector with the pinned ARM prompt and durable usage log."""
import argparse
import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openwebrl.arm_inference import load_selection_builder, selection_messages, parse_selection

MODEL='gpt-5.6-sol'
# USD per million tokens: input, cached input, output (gpt-6-luna as in openwebrl/reasoning_actor_policy.py).
# gpt-6-sol: list price not recorded in this repo; accounted at gpt-6.1-sol's (openwebrl/reasoning_actor_policy.py).
# Raw token usage is logged per request, so spend can be re-priced exactly.
PRICES={'gpt-5.6-sol':(4.,.4,20.),'gpt-6-luna':(.10,.01,.50),'gpt-6-sol':(2.,.10,10.)}
SOURCE=Path('/gpfs/scrubbed/zixianma/openwebrl-runtime/arm-reproduction/source')


class Request(BaseModel):
    mode: str
    task: str
    url: str
    history: list[dict]
    candidates: list[dict]
    screenshot: str
    candidate_representation: str='full'


def response_input(request,builder):
    if request.mode!='selection' or request.candidate_representation!='full' or not 1<=len(request.candidates)<=5:
        raise ValueError('Only full-representation ARM selection is supported')
    image=base64.b64decode(request.screenshot,validate=True)
    messages=selection_messages(builder,request.task,request.url,request.history,request.candidates,image)
    result=[]
    for message in messages:
        content=message['content']
        if isinstance(content,str): content=[dict(type='text',text=content)]
        pieces=[]
        for part in content:
            if part['type']=='text': pieces.append(dict(type='input_text',text=part['text']))
            elif part['type']=='image_url':
                pieces.append(dict(type='input_image',image_url=part['image_url']['url'],detail='high'))
            else: raise ValueError('Unexpected ARM prompt modality')
        result.append(dict(role=message['role'],content=pieces))
    return result


class Selector:
    def __init__(self,args,client):
        self.args=args; self.client=client; self.builder=load_selection_builder(SOURCE)
        self.semaphore=asyncio.Semaphore(args.concurrency)
        self.calls=0; self.cost=0.; self.reserved=0.; self.consecutive_failures=0
        self.root=Path(args.output); self.root.mkdir(parents=True,exist_ok=True)
        self.log=self.root/'api-requests.jsonl'
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                row=json.loads(line); self.calls+=1; self.cost+=row['accounted_cost_usd']

    def health(self):
        return dict(mode='selection',model=self.args.model,provider='openai-responses',reasoning_effort=self.args.reasoning_effort,
            max_output_tokens=self.args.max_output_tokens,image_detail='high',source_root=str(SOURCE),
            selection_decoding='strict_json_schema',candidate_representation='full')

    async def select(self,request: Request):
        inputs=response_input(request,self.builder)
        # Conservative reservation includes image tokens and upper bound of one
        # input token per UTF-8 byte. Reserve output including hidden reasoning.
        k=len(request.candidates)
        properties={'selection':{'type':'integer','enum':list(range(1,k+1))}}
        if self.args.probabilities:
            # Prompted confidence: same pinned ARM prompt, plus a request for a probability per candidate.
            inputs[-1]['content'].append(dict(type='input_text',text=(
                f'In addition to "selection", give "p1".."p{k}": your probability (0 to 1, summing to 1) that each '
                'candidate is the best next action. The selected candidate must have the highest probability.')))
            properties.update({f'p{i}':{'type':'number'} for i in range(1,k+1)})
        text_bytes=sum(len(p.get('text','').encode()) for m in inputs for p in m['content'])
        pin,_,pout=PRICES[self.args.model]
        reserve=(text_bytes+8192)*pin/1e6+self.args.max_output_tokens*pout/1e6
        async with self.semaphore:
            if self.calls>=self.args.max_requests or self.cost+self.reserved+reserve>self.args.max_cost_usd:
                (self.root/'budget-exhausted.json').write_text(json.dumps(dict(
                    requests=self.calls,accounted_cost_usd=self.cost,reserved_cost_usd=self.reserved)))
                raise HTTPException(429,'Selector run request/cost cap reached')
            self.calls+=1; self.reserved+=reserve
            started=time.monotonic()
            record=dict(request_number=self.calls,task=request.task,
                payload_sha256=hashlib.sha256(request.model_dump_json().encode()).hexdigest(),
                requested_model=self.args.model,accounted_cost_usd=reserve,passed=False)
            try:
                result=await self.client.responses.create(model=self.args.model,input=inputs,
                    reasoning=dict(effort=self.args.reasoning_effort),max_output_tokens=self.args.max_output_tokens,
                    text={'format':{'type':'json_schema','name':'action_selection','strict':True,'schema':{
                        'type':'object','properties':properties,
                        'required':list(properties),'additionalProperties':False}}})
                usage=result.usage.model_dump()
                cached=usage.get('input_tokens_details',{}).get('cached_tokens',0)
                pin,pcached,pout=PRICES[self.args.model]
                cost=((usage['input_tokens']-cached)*pin+cached*pcached+usage['output_tokens']*pout)/1e6
                record.update(response_id=result.id,returned_model=result.model,status=result.status,
                    raw=result.output_text,usage=usage,accounted_cost_usd=cost)
                if not result.model.startswith(self.args.model) or result.status!='completed': raise ValueError('Incomplete or substituted selector response')
                scores=None; raw=result.output_text
                if self.args.probabilities:
                    values=json.loads(result.output_text); selection=int(values['selection'])
                    if not 1<=selection<=k: raise ValueError('selection out of range')
                    raw_p=[max(0.,float(values[f'p{i}'])) for i in range(1,k+1)]
                    total=sum(raw_p); scores=[v/total for v in raw_p] if total>0 else None
                    raw=json.dumps({'selection':selection})   # canonical form for downstream parse_selection
                parse_selection(raw,k)
                record['passed']=True
                return dict(raw=raw,scores=scores,candidate_representation='full',model=result.model,
                    input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],
                    reasoning_tokens=usage.get('output_tokens_details',{}).get('reasoning_tokens',0),
                    cost_usd=cost,seconds=time.monotonic()-started,response_id=result.id)
            except Exception as exc:
                record.update(error_type=type(exc).__name__,http_status=getattr(exc,'status_code',None),error_code=getattr(exc,'code',None))
                raise HTTPException(502,'Selector response unavailable; inspect durable API telemetry') from None
            finally:
                self.reserved-=reserve; self.cost+=record['accounted_cost_usd']
                record.update(seconds=time.monotonic()-started,total_accounted_cost_usd=self.cost)
                self.consecutive_failures=0 if record['passed'] else self.consecutive_failures+1
                with self.log.open('a') as f: f.write(json.dumps(record,ensure_ascii=False)+'\n')
                path=self.root/'usage.json'; temporary=path.with_suffix('.partial')
                temporary.write_text(json.dumps(dict(requests=self.calls,accounted_cost_usd=self.cost,
                    reserved_cost_usd=self.reserved,max_cost_usd=self.args.max_cost_usd,
                    consecutive_failures=self.consecutive_failures)))
                temporary.replace(path)


def app(args,client=None):
    from openai import AsyncOpenAI
    client=client or AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'],timeout=100,max_retries=0)
    selector=Selector(args,client); api=FastAPI()
    api.get('/health')(selector.health); api.post('/select')(selector.select)
    return api


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port',type=int,default=27511)
    p.add_argument('--model',choices=sorted(PRICES),default=MODEL)
    p.add_argument('--probabilities',action='store_true',help='also ask for a probability per candidate (returned as scores)')
    p.add_argument('--output',required=True)
    p.add_argument('--concurrency',type=int,default=16)
    p.add_argument('--reasoning-effort',choices=['medium','high'],default='medium')
    p.add_argument('--max-output-tokens',type=int,default=2048)
    p.add_argument('--max-requests',type=int,default=10000)
    p.add_argument('--max-cost-usd',type=float,default=200)
    return p


if __name__=='__main__':
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1]/'.env',override=False)
    a=parser().parse_args()
    uvicorn.run(app(a),host='127.0.0.1',port=a.port)
