import gc
import hashlib
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from kairos.data import load_posts,load_labels,load_followers
from kairos.label import author_level,fit_cutoffs
from solution.content import (enrich,normalized_times,extra_content,content,aligned,_IDENTITY,
    past_semantics,past_vectors,history_indices,cdf_features,visual_history,GENRES)


def file_hash(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save_array(path,values):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.save(stream,values)
    temporary.replace(path)


def cached_resource(path,configuration,spec=None):
    status=path.with_suffix('.status.json')
    if status.exists():
        saved=json.loads(status.read_text())
        if (saved['configuration']!=spec or saved.get('status')!='complete'
                or saved.get('sha256')!=file_hash(path)):
            raise ValueError('Cached encoder resource changed: '+str(path))
    elif configuration['legacy_cache_sha256'].get(path.name)!=file_hash(path):
        raise ValueError('Cached resource is not from the selected version: '+str(path))


def encode(root,texts,info,path,dimension,batch_size=32,skip_empty=False,order=None):
    import torch
    from sentence_transformers import SentenceTransformer
    signatures=hashlib.sha256()
    for text in texts:signatures.update((text+'\n').encode())
    spec={'model':info,'dimensions':dimension,'input_sha256':signatures.hexdigest(),
          'rows':len(texts),'max_length':384,'skip_empty':skip_empty,'batch_size':batch_size}
    status=path.with_suffix('.status.json')
    completed=0
    if status.exists():
        old=json.loads(status.read_text())
        if old['configuration']!=spec:raise ValueError('Encoder inputs changed: '+str(path))
        if old['status']=='complete':
            if old.get('sha256')!=file_hash(path):raise ValueError('Completed text vectors changed')
            return np.load(path,mmap_mode='r')
        completed=old['completed']
    values=np.lib.format.open_memmap(path,mode='r+' if status.exists() else 'w+',dtype='float32',shape=(len(texts),dimension))
    if not status.exists():
        values[:]=0;values.flush()
        status.write_text(json.dumps({'configuration':spec,'completed':0,'status':'running'})+'\n')
    device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    model=SentenceTransformer(info['name'],revision=info['revision'],device=device,
        cache_folder=str(root/'.cache/huggingface/hub'),trust_remote_code=False,token=False)
    model.max_seq_length=384
    order=np.arange(len(texts)) if order is None else order
    started=last=time.monotonic()
    for start in range(completed,len(order),256):
        rows=order[start:start+256]
        if skip_empty:rows=np.asarray([i for i in rows if texts[i].strip()])
        if len(rows):
            values[rows]=model.encode(['query: '+texts[i] for i in rows],batch_size=batch_size,
                normalize_embeddings=True,convert_to_numpy=True,precision='float32',show_progress_bar=False)
        values.flush();done=min(start+256,len(order))
        state={'configuration':spec,'completed':done,'status':'complete' if done==len(order) else 'running'}
        if done==len(order):state['sha256']=file_hash(path)
        status.write_text(json.dumps(state)+'\n')
        if time.monotonic()-last>30 or done==len(order):
            print('ENCODE',info['name'],path.name,done,len(order),'seconds',round(time.monotonic()-started),flush=True)
            last=time.monotonic()
    del model;gc.collect()
    if device=='mps':torch.mps.empty_cache()
    return values


def text_resources(root,work,posts,attachments,public,configuration):
    keys=posts[['author','post']]
    own=(posts.text+'\n'+attachments.attached_text).str.replace(_IDENTITY,r'⟨\1⟩',regex=True)
    expanded=own.copy();extra=public.extra_public_text.str.replace(_IDENTITY,r'⟨\1⟩',regex=True)
    present=extra.str.strip().ne('');expanded.loc[present]+='\n'+extra[present]
    body=(posts.text+'\n').str.replace(_IDENTITY,r'⟨\1⟩',regex=True)
    context='\n'+attachments.attached_text
    context.loc[present]+='\n'+extra[present]
    context=context.str.replace(_IDENTITY,r'⟨\1⟩',regex=True)
    inputs=keys.assign(rich_text=own)
    original=work/'embedding-posts.pkl'
    if original.exists():pd.testing.assert_frame_equal(pd.read_pickle(original),inputs)
    else:inputs.to_pickle(original)
    expanded_inputs=keys.assign(rich_text=expanded)
    if (work/'extended-posts.pkl').exists():pd.testing.assert_frame_equal(pd.read_pickle(work/'extended-posts.pkl'),expanded_inputs)
    else:expanded_inputs.to_pickle(work/'extended-posts.pkl')
    roles=keys.assign(body_text=body,context_text=context)
    if (work/'role-inputs.pkl').exists():pd.testing.assert_frame_equal(pd.read_pickle(work/'role-inputs.pkl'),roles)
    else:roles.to_pickle(work/'role-inputs.pkl')
    small_path=work/'embeddings.npy'
    if small_path.exists() and not small_path.with_suffix('.status.json').exists():cached_resource(small_path,configuration)
    small=(np.load(small_path,mmap_mode='r') if small_path.exists() and not small_path.with_suffix('.status.json').exists() else
        encode(root,own.tolist(),configuration['encoder_models']['small'],small_path,384))
    base_path=work/'backbone-embeddings.npy'
    if base_path.exists() and not base_path.with_suffix('.status.json').exists():cached_resource(base_path,configuration)
    base=(np.load(base_path,mmap_mode='r') if base_path.exists() and not base_path.with_suffix('.status.json').exists() else
        encode(root,expanded.tolist(),configuration['encoder_models']['base'],base_path,768,16,
               order=np.random.default_rng(20261024).permutation(len(posts))))
    role_path=work/'role-embeddings.npy'
    role_spec={'model':configuration['encoder_models']['small'],'input_sha256':file_hash(work/'role-inputs.pkl')}
    if role_path.exists():
        cached_resource(role_path,configuration,role_spec)
        separated=np.load(role_path,mmap_mode='r')
    else:
        temporary=work/'role-embeddings.partial.npy'
        separated=np.lib.format.open_memmap(temporary,mode='w+',dtype='float32',shape=(len(posts),2,384))
        for number,values in enumerate([body,context]):
            separated[:,number]=encode(root,values.tolist(),configuration['encoder_models']['small'],
                work/f'role-{number}.npy',384,skip_empty=True)
        separated.flush();temporary.replace(role_path)
        role_path.with_suffix('.status.json').write_text(json.dumps({'configuration':role_spec,'status':'complete','sha256':file_hash(role_path)})+'\n')
    for values,shape in [(small,(len(posts),384)),(base,(len(posts),768)),(separated,(len(posts),2,384))]:
        if values.shape!=shape or not np.isfinite(values).all():raise ValueError('Invalid cached text embeddings')
    for values in [small,base]:np.testing.assert_allclose(np.linalg.norm(values,axis=1),1,atol=1e-4,rtol=0)
    role_norms=np.linalg.norm(separated,axis=2)
    for number,values in enumerate([body,context]):
        np.testing.assert_allclose(role_norms[:,number],values.str.strip().ne('').astype(float),atol=1e-4,rtol=0)
    return small,base,separated


def image_selection(data,work,labels):
    path=work/'image-selection.json'
    if path.exists():return json.loads(path.read_text())
    manifest=data/'media/manifest.jsonl'
    if not manifest.exists():manifest=work/'media-manifest.jsonl'
    if not manifest.exists():raise FileNotFoundError('Unpack media/manifest.jsonl from the core archive')
    keys=set(map(tuple,labels[['author','post']].to_numpy()))
    priority={'post_image':0,'article_image':1,'document_cover':2,'celebration_image':3,'video_poster':4}
    selected={}
    for line in manifest.open():
        row=json.loads(line);key=row['author'],row['post'];kind=row['kind']
        if key not in keys or kind not in priority or not row.get('content_type','').startswith('image/'):continue
        if row['bytes']>25*1024**2:continue
        if key not in selected or priority[kind]<selected[key]['priority']:
            selected[key]={**{k:row[k] for k in ['author','post','bytes','kind']},
                           'file':'media/'+row['file'],'priority':priority[kind]}
    items=list(selected.values());files={x['file']:x['bytes'] for x in items}
    result={'selection':'first content image per labelled post, labels not consulted','post_images':items,
            'unique_files':files,'bytes':sum(files.values())}
    path.write_text(json.dumps(result,indent=2)+'\n');return result


def image_resources(root,data,work,labels,configuration):
    selection=image_selection(data,work,labels);files=sorted(selection['unique_files'])
    path=work/'image-embeddings.npy';metadata_path=work/'image-metadata.jsonl'
    image_spec={'model':configuration['encoder_models']['clip'],'selection_sha256':file_hash(work/'image-selection.json')}
    if path.exists() and metadata_path.exists():
        cached_resource(path,configuration,image_spec)
        if path.with_suffix('.status.json').exists():
            if json.loads(path.with_suffix('.status.json').read_text())['metadata_sha256']!=file_hash(metadata_path):
                raise ValueError('Image metadata changed')
        else:
            cached_resource(metadata_path,configuration)
            cached_resource(work/'image-selection.json',configuration)
        vectors=np.load(path,mmap_mode='r');metadata=pd.read_json(metadata_path,lines=True)
        if vectors.shape!=(len(files),512) or metadata.file.tolist()!=files:raise ValueError('Incomplete image cache')
        return selection,vectors,metadata
    import torch
    from PIL import Image,ImageOps
    from sentence_transformers import SentenceTransformer
    Image.MAX_IMAGE_PIXELS=16_000_000;torch.set_num_threads(2)
    info=configuration['encoder_models']['clip']
    model=SentenceTransformer(info['name'],revision=info['revision'],device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
        cache_folder=str(root/'.cache/huggingface/hub'),trust_remote_code=False,token=False)
    vectors=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(len(files),512));vectors[:]=0
    records=[]
    for start in range(0,len(files),32):
        images,positions=[],[]
        for number in range(start,min(start+32,len(files))):
            name=files[number];image_path=data/name.removeprefix('media/') if data.name=='media' else data/name
            if not image_path.is_file():image_path=root/'.tmp/content-images'/name
            if not image_path.is_file():raise FileNotFoundError('Unpack selected media before encoding: '+name)
            row={'file':name,'valid':False,'bytes':selection['unique_files'][name]}
            try:
                if file_hash(image_path)!=image_path.stem:raise ValueError('Content hash mismatch')
                with warnings.catch_warnings():
                    warnings.simplefilter('error',Image.DecompressionBombWarning)
                    with Image.open(image_path) as image:
                        width,height=image.size
                        if width*height>Image.MAX_IMAGE_PIXELS:raise ValueError('Image exceeds pixel budget')
                        converted=ImageOps.exif_transpose(image).convert('RGB')
                images.append(converted);positions.append(number);row.update(valid=True,width=width,height=height)
            except (OSError,ValueError,Image.DecompressionBombError,Image.DecompressionBombWarning) as exc:row['error']=str(exc)
            records.append(row)
        if images:
            vectors[positions]=model.encode(images,batch_size=32,normalize_embeddings=True,
                convert_to_numpy=True,precision='float32',show_progress_bar=False)
        for image in images:image.close()
        if start%1024==0:print('CLIP_IMAGES',start,len(files),flush=True)
    vectors.flush();pd.DataFrame(records).to_json(metadata_path,orient='records',lines=True)
    path.with_suffix('.status.json').write_text(json.dumps({'configuration':image_spec,'status':'complete',
        'sha256':file_hash(path),'metadata_sha256':file_hash(metadata_path)})+'\n')
    del model;gc.collect()
    return selection,vectors,pd.DataFrame(records)


def visual_features(root,data,work,table,posts,labels,configuration):
    selection,vectors,metadata=image_resources(root,data,work,labels,configuration)
    files=sorted(selection['unique_files']);selected=pd.DataFrame(selection['post_images'])
    ids=pd.Index(files).get_indexer(selected.file)
    joins=table[['author','post']].merge(selected.assign(image_row=ids),on=['author','post'],how='left',validate='one_to_one',sort=False)
    rows=joins.image_row.fillna(-1).to_numpy(dtype=int);exists=rows>=0
    current=np.zeros((len(table),512),dtype='float32');current[exists]=vectors[rows[exists]]
    has_image=np.linalg.norm(current,axis=1)>.5
    geometry=pd.DataFrame(index=table.index)
    geometry['has_content_image']=has_image.astype('float32');geometry['content_image_invalid']=(exists&~has_image).astype('float32')
    geometry['image_log_bytes']=np.log1p(joins['bytes'].fillna(0)).astype('float32')
    width,height=np.zeros(len(table)),np.zeros(len(table))
    width[exists]=metadata.width.fillna(0).to_numpy()[rows[exists]];height[exists]=metadata.height.fillna(0).to_numpy()[rows[exists]]
    geometry['image_log_pixels']=np.log1p(width*height).astype('float32')
    geometry['image_aspect_ratio']=np.divide(width,height,out=np.zeros(len(table)),where=height>0).astype('float32')
    for kind in ['post_image','article_image','document_cover','celebration_image','video_poster']:
        geometry['image_kind_'+kind]=joins.kind.eq(kind).to_numpy(dtype='float32')
    projection=np.random.default_rng(932).normal(0,1/8,(512,64)).astype('float32')
    lookup=pd.MultiIndex.from_frame(table[['author','post']])
    positions=lookup.get_indexer(pd.MultiIndex.from_frame(posts[['author','post']]))
    all_vectors=np.zeros((len(posts),512),dtype='float32');known=positions>=0;all_vectors[known]=current[positions[known]]
    with threadpool_limits(limits=2):
        history=visual_history(posts,all_vectors)
        projected=current@projection
    direction_path=work/'visual-topic-directions.npy'
    direction_spec={'model':configuration['encoder_models']['clip'],'names':list(GENRES),'descriptions':list(GENRES.values())}
    if direction_path.exists():
        cached_resource(direction_path,configuration,direction_spec)
        directions=np.load(direction_path)
    else:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(2);info=configuration['encoder_models']['clip']
        model=SentenceTransformer(info['name'],revision=info['revision'],device='cpu',cache_folder=str(root/'.cache/huggingface/hub'),trust_remote_code=False,token=False)
        prompts=[template.format(description) for description in GENRES.values() for template in
                 ['An image showing {}.','A social media post with {}.','A photograph or graphic showing {}.']]
        encoded=model.encode(prompts,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,precision='float32',show_progress_bar=False)
        directions=encoded.reshape(len(GENRES),3,512).mean(1);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        save_array(direction_path,directions)
        direction_path.with_suffix('.status.json').write_text(json.dumps({'configuration':direction_spec,'status':'complete',
            'sha256':file_hash(direction_path)})+'\n')
        del model;gc.collect()
    with threadpool_limits(limits=2):scores=current@directions.T
    return pd.concat([geometry,pd.DataFrame(projected,columns=[f'image_projection_{i}' for i in range(64)],index=table.index),
        aligned(history,table),pd.DataFrame(scores,columns=['visual_topic_'+name for name in GENRES],index=table.index)],axis=1)


def prepare(root,data,cache,configuration):
    cache.mkdir(parents=True,exist_ok=True);work=root/'work';work.mkdir(exist_ok=True)
    labels=load_labels(data)[['author','post','split','label']]
    labels.loc[labels.split.eq('test'),'label']=np.nan
    posts=author_level(load_posts(data)).reset_index(drop=True)
    attachments=aligned(extra_content(data),posts).reset_index(drop=True)
    public=aligned(content(data),posts).reset_index(drop=True)
    small,base,roles=text_resources(root,work,posts,attachments,public,configuration)
    common=enrich(normalized_times(posts),load_followers(data))
    for name in attachments:common[name]=attachments[name].to_numpy()
    for name in public:common[name]=public[name].to_numpy()
    common['rich_text']=(common.text+'\n'+common.attached_text).str.replace(_IDENTITY,r'⟨\1⟩',regex=True)+'\n'+common.extra_public_text.str.replace(_IDENTITY,r'⟨\1⟩',regex=True)
    common['attached_chars']=common.attached_text.str.len()
    table=labels.merge(common,on=['author','post'],how='left',validate='one_to_one',sort=False)
    if len(table)!=89018 or table.level.isna().any():raise ValueError('Release population differs from the selected solution')
    numeric=[name for name in table if pd.api.types.is_numeric_dtype(table[name]) and name!='label']
    table[numeric]=table[numeric].replace([np.inf,-np.inf],np.nan).astype('float32')
    for name in ['content_type','previous_content_type','reshared_content_type']:table[name]=table[name].fillna('unknown').astype(str)
    table=pd.concat([table,visual_features(root,data,work,table,posts,labels,configuration)],axis=1)
    keys=posts[['author','post']]
    def embedding_block(vectors,prefix,history_prefix):
        similarity,delta=past_vectors(keys,vectors) if vectors.shape[1]==768 else past_semantics(keys,vectors)
        names=[f'{prefix}_embedding_{i}' for i in range(vectors.shape[1])]
        names += [f'{history_prefix}_past_cosine_{w}' for w in [1,5,20]]+[f'{history_prefix}_past_delta_{i}' for i in range(64)]
        return aligned(pd.concat([keys,pd.DataFrame(np.column_stack([vectors,similarity,delta]),columns=names)],axis=1),table)
    backbone=pd.concat([table,embedding_block(base,'backbone','backbone')],axis=1)
    role_blocks=[embedding_block(roles[:,i],name,name) for i,name in enumerate(['body','context'])]
    role_table=pd.concat([table,*role_blocks],axis=1)
    role_inputs=pd.read_pickle(work/'role-inputs.pkl')
    flags=keys.assign(has_body_text=role_inputs.body_text.str.strip().ne('').astype('float32'),
        has_context_text=role_inputs.context_text.str.strip().ne('').astype('float32'),body_context_cosine=np.einsum('ij,ij->i',roles[:,0],roles[:,1]))
    role_table=pd.concat([role_table,aligned(flags,table)],axis=1)
    cutoffs=fit_cutoffs(posts.loc[posts.author.isin(set(labels.loc[labels.split.eq('train'),'author'])),'resid'])
    cdf=aligned(cdf_features(posts,cutoffs),table)
    tables={'time-v2':table,'backbone-v3':backbone,'roles-v3':role_table,'roles-cdf-v3':pd.concat([role_table,cdf],axis=1)}
    for item in configuration['components']:
        name=item['cache'];schema=item['schema'];feature=item['config']['features']
        columns=['author','post','split','label',*schema[feature],*schema[feature+'_categorical']]
        if item['config'].get('include_text'):columns.append('rich_text')
        selected=tables[name][list(dict.fromkeys(columns))].copy()
        path=cache/(name+'.pkl')
        if path.exists():
            old=pd.read_pickle(path)
            for column in selected:
                if pd.api.types.is_numeric_dtype(selected[column]):
                    np.testing.assert_allclose(selected[column],old[column],rtol=1e-5,atol=1e-6,equal_nan=True)
                else:pd.testing.assert_series_equal(selected[column],old[column],check_dtype=False)
            print('FEATURES_MATCH_EXISTING',name,len(selected),flush=True)
        else:selected.to_pickle(path);print('FEATURES_WRITTEN',name,len(selected),flush=True)
    indices,ages=history_indices(posts)
    history=dict(indices=indices,ages=ages,log_reactions=posts.log_reactions.to_numpy(dtype='float32'),
        level=posts.level.fillna(0).to_numpy(dtype='float32'),content_type=pd.factorize(posts.content_type)[0].astype('int32'))
    history_path=work/'round3-history.npz'
    if history_path.exists():
        with np.load(history_path) as previous:
            for name,values in history.items():np.testing.assert_array_equal(values,previous[name])
    else:np.savez_compressed(history_path,**history)
    history_keys=work/'round3-history-posts.pkl'
    if history_keys.exists():pd.testing.assert_frame_equal(keys,pd.read_pickle(history_keys))
    else:keys.to_pickle(history_keys)
    print('SELECTED_FEATURES_READY',flush=True)
