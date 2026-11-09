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
        return engine.get_text_pe(names), {'status':'disabled'}
    root=Path(__file__).resolve().parent
    directory=Path(directory) if directory else root/'outputs'/'text_embeddings'
    directory=directory.resolve()
    if not directory.is_relative_to(root):
        raise ValueError('Embedding cache must stay inside lan_chiem')
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
    embeddings=engine.get_text_pe(names)
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
