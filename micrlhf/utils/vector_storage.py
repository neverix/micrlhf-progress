import jax.numpy as jnp
from penzai import pz

from typing import Union
from pathlib import Path

from appdirs import *
appname = "micrlhf"
appauthor = "nev"

SAVE_EXT = ".npy"
CACHE_DIR = user_cache_dir(appname, appauthor)
REPO_NAME = "dmitriihook/micrlhf"


def save_vector(name: str, vector: Union[jnp.ndarray, pz.nx.NamedArray], overwrite: bool = False):    
    save_path = Path(CACHE_DIR) / name
    save_path = save_path.with_suffix(SAVE_EXT)

    if not overwrite and save_path.exists():
        raise FileExistsError(f'{save_path} already exists. Set overwrite=True to overwrite the file.')
    
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    
    if isinstance(vector, jnp.ndarray):
        jnp.save(save_path, vector)

    elif isinstance(vector, pz.nx.NamedArray):
        jnp.save(save_path, vector)

def load_vector(name: str) -> Union[jnp.ndarray, pz.nx.NamedArray]:
    load_path = Path(CACHE_DIR) / name
    load_path = load_path.with_suffix(SAVE_EXT)

    if not load_path.exists():
        raise FileNotFoundError(f'{load_path} does not exist.')

    return jnp.load(load_path)