"""Safe, versioned NumPy cache of YOLOE's constant text embeddings."""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4
from zipfile import BadZipFile
import numpy as np
import torch


def text_embeddings(engine, names, device, revision, vocabulary_hash, versions, enabled=True, directory=None):
    if not enabled or revision is None:
        return _compute_text_embeddings(engine, names, device), {'status':'disabled'}
    from uav_pipeline.paths import PROJECT_ROOT, CACHE_DIR
    root=PROJECT_ROOT
    directory=Path(directory) if directory else CACHE_DIR/'text_embeddings'
    directory=directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Embedding cache must stay inside the UAV project')
    key=hashlib.sha256(json.dumps({'schema':1,'weights':revision,'vocabulary':vocabulary_hash,
                                  'versions':versions},sort_keys=True).encode()).hexdigest()
    path=directory/f'{key}.npz'
    dimension=engine.model.model[-1].embed
    if not isinstance(dimension,int) or dimension < 1:
        raise ValueError('YOLOE head must expose its embedding dimension')
    expected=(1,len(names),dimension)
    if path.is_file():
        try:
            with np.load(path,allow_pickle=False) as saved:
                array=saved['embeddings']
                if str(saved['cache_key'].item())==key and array.shape==expected and array.dtype==np.float32 and np.isfinite(array).all():
                    return torch.from_numpy(array.copy()).to(device), {'status':'hit','key':key,'path':str(path)}
        except (OSError,ValueError,KeyError,EOFError,BadZipFile):
            pass  # Interrupted/invalid cache is recomputed from the checkpoint.
    embeddings=_compute_text_embeddings(engine, names, device)
    array=embeddings.detach().float().cpu().numpy()
    if array.shape!=expected or not np.isfinite(array).all():
        raise RuntimeError('Invalid YOLOE text embeddings')
    directory.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f'.{key}-{uuid4().hex}.tmp')
    try:
        with temporary.open('wb') as file:
            np.savez_compressed(file,embeddings=array,cache_key=np.array(key))
        os.replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()
    return embeddings, {'status':'computed','key':key,'path':str(path)}


def _compute_text_embeddings(engine, names, device):
    """YOLOE-26 uses its local encoder asset; fused v8 never enters this path."""
    from uav_pipeline.paths import MODELS_DIR
    model = getattr(engine, 'model', None)
    encoder_path = MODELS_DIR/'mobileclip2_b.ts'
    variant = getattr(model, 'text_model', '')
    if isinstance(model, torch.nn.Module) and isinstance(variant, str) and variant.startswith('mobileclip2:') and encoder_path.is_file():
        from ultralytics.nn.text_model import MobileCLIPTS
        previous = getattr(model, 'clip_model', None)
        model.clip_model = MobileCLIPTS(device=torch.device(device), weight=str(encoder_path))
        try:
            return model.get_text_pe(names, cache_clip_model=True)
        finally:
            if previous is None:
                del model.clip_model
            else:
                model.clip_model = previous
    return engine.get_text_pe(names)
