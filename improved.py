"""Selected Lab1 solution. Run python improved.py --help for reproduction commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
for key,suffix in {'HF_HOME':'.cache/huggingface','TORCH_HOME':'.cache/torch','XDG_CACHE_HOME':'.cache',
                   'MPLCONFIGDIR':'.cache/matplotlib','TMPDIR':'.tmp'}.items():
    os.environ[key]=str(ROOT/suffix);(ROOT/suffix).mkdir(parents=True,exist_ok=True)
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO','0.4')
os.environ.setdefault('PYTORCH_MPS_LOW_WATERMARK_RATIO','0.2')

import joblib
import numpy as np
import pandas as pd
from solution.train import load_frames,train


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def predict(bundle,frames,keys):
    requested=pd.MultiIndex.from_frame(keys[['author','post']])
    if not requested.is_unique or keys.isna().any().any():raise ValueError('Prediction keys must be unique and present')
    selected={}
    for name,frame in frames.items():
        positions=pd.MultiIndex.from_frame(frame[['author','post']]).get_indexer(requested)
        if (positions<0).any():raise ValueError('Requested post has no prepared features')
        selected[name]=frame.iloc[positions]
    labels=bundle.predict(selected)
    if labels.shape!=(len(keys),) or not np.isin(labels,[0,1,2]).all():raise ValueError('Invalid predictions')
    return keys[['author','post']].assign(label=labels)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',choices=['prepare','train','predict'],default='predict')
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'work',help='Feature tables; encoder/history resources remain in work/')
    parser.add_argument('--model-dir',type=Path,default=ROOT/'models/best')
    parser.add_argument('--output',type=Path,default=ROOT/'lab1_submission')
    parser.add_argument('--keys',type=Path,help='Optional author/post CSV; otherwise predict the official test keys')
    args=parser.parse_args()
    config_path=ROOT/'solution/config.json';configuration=json.loads(config_path.read_text())
    if args.command=='prepare':
        from solution.prepare import prepare
        prepare(ROOT,args.data,args.cache_dir,configuration)
        return
    missing=[c['cache'] for c in configuration['components'] if not (args.cache_dir/(c['cache']+'.pkl')).exists()]
    if missing:
        raise SystemExit('Prepare features first: python improved.py prepare --data '+str(args.data)+' --cache-dir '+str(args.cache_dir))
    frames=load_frames(args.cache_dir,configuration)
    if args.command=='train':
        bundle,report=train(configuration,frames,args.model_dir)
        report.update(model_sha256=digest(args.model_dir/'model.joblib'),config_sha256=digest(config_path))
        (args.model_dir/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    else:
        model_path=args.model_dir/'model.joblib'
        if not model_path.exists():raise SystemExit('Train the fixed solution first: python improved.py train --model-dir '+str(args.model_dir))
        record_path=args.model_dir/'selection.json'
        if record_path.exists():
            record=json.loads(record_path.read_text())
            if digest(model_path)!=record['model_sha256'] or digest(config_path)!=record['config_sha256']:
                raise ValueError('Saved model/configuration differs from its selected version')
        bundle=joblib.load(model_path)
    reference=next(iter(frames.values()))
    keys=(pd.read_csv(args.keys,usecols=['author','post']) if args.keys else
          reference.loc[reference.split.eq('test'),['author','post']])
    result=predict(bundle,frames,keys)
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'predictions.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path),result.reset_index(drop=True))
        print('Existing predictions match exactly:',len(result),path,flush=True)
    else:
        result.to_csv(path,index=False);print('Predictions written:',len(result),path,flush=True)
    print('SHA256:',digest(path),flush=True)


if __name__=='__main__':main()
