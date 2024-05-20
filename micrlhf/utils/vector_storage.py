import jax.numpy as jnp
import numpy as np

from collections import OrderedDict
from huggingface_hub import HfApi
from huggingface_hub import HfFileSystem
from penzai import pz

from typing import Union
from pathlib import Path

from appdirs import *


appname = "micrlhf"
appauthor = "nev"

SAVE_EXT = ".npz"
CACHE_DIR = user_cache_dir(appname, appauthor)
REPO_NAME = "kisate-team/micrlhf"
REPO_PATH_PREFIX = Path("vectors")


def save_vector(name: str, vector: Union[jnp.ndarray, pz.nx.NamedArray], overwrite: bool = False):    
    save_path = Path(CACHE_DIR) / name
    save_path = save_path.with_suffix(SAVE_EXT)

    if not overwrite and save_path.exists():
        raise FileExistsError(f'{save_path} already exists. Set overwrite=True to overwrite the file.')
    
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    
    if isinstance(vector, jnp.ndarray):
        jnp.savez(save_path, data = vector)

    elif isinstance(vector, pz.nx.NamedArray):
        # Does not support both positional and named axes

        shape = vector.named_shape
        vector = vector.unwrap(*shape.keys())
        jnp.savez(save_path, data = vector, shape = shape)

def upload_vector(name: str, overwrite: bool = False):
    save_path = Path(CACHE_DIR) / name
    save_path = save_path.with_suffix(SAVE_EXT)

    if not save_path.exists():
        raise FileNotFoundError(f'{save_path} does not exist.')

    repo_path = REPO_PATH_PREFIX / name
    repo_path = repo_path.with_suffix(SAVE_EXT)

    if not overwrite:
        fs = HfFileSystem()
        if fs.exists(repo_path):
            raise FileExistsError(f'{repo_path} already exists in the remote repository. Set overwrite=True to overwrite the file.')
    
    api = HfApi()
    api.upload_file(
        path_or_fileobj=save_path,
        repo_id=REPO_NAME,
        path_in_repo= str(repo_path),
        repo_type="model"
    )

def save_and_upload_vector(name: str, vector: Union[jnp.ndarray, pz.nx.NamedArray], overwrite: bool = False):
    save_vector(name, vector, overwrite=overwrite)
    upload_vector(name, overwrite=overwrite)

def load_vector_from_path(load_path: Path) -> Union[jnp.ndarray, pz.nx.NamedArray]:
    if not load_path.exists():
        raise FileNotFoundError(f'{load_path} does not exist.')

    data = np.load(load_path, allow_pickle=True)

    if 'shape' in data:
        shape = data['shape']
        data = data['data']

        if data.dtype == 'V2':
            data = data.view(jnp.bfloat16)
        
        data = jnp.asarray(data)
        return pz.nx.NamedArray(OrderedDict(shape.tolist()), data)

    return jnp.load(load_path)

def download_vector(name: str, overwrite: bool = False) -> Union[jnp.ndarray, pz.nx.NamedArray]:
    repo_path = Path(REPO_NAME) / REPO_PATH_PREFIX / name
    repo_path = repo_path.with_suffix(SAVE_EXT)
    save_path = Path(CACHE_DIR) / name
    save_path = save_path.with_suffix(SAVE_EXT)

    fs = HfFileSystem()


    if not fs.exists(repo_path):
        raise FileNotFoundError(f'{repo_path} does not exist in the remote repository.')

    if not overwrite and save_path.exists():
        raise FileExistsError(f'{save_path} already exists. Set overwrite=True to overwrite the file.')

    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'wb') as fi:
        with fs.open(repo_path, 'rb') as fo:
            fi.write(fo.read())

    return load_vector_from_path(save_path)

def load_vector(name: str, from_remote=False, overwrite=False) -> Union[jnp.ndarray, pz.nx.NamedArray]:
    load_path = Path(CACHE_DIR) / name
    load_path = load_path.with_suffix(SAVE_EXT)

    if not load_path.exists():
        from_remote = True

    if from_remote:
        return download_vector(name, overwrite=overwrite)

    return load_vector_from_path(load_path)
