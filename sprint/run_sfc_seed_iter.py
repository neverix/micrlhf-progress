import subprocess

tasks = ["antonyms", "en_es", "country_capital", "present_gerund"]

seeds = list(range(10, 30))

from tqdm.auto import tqdm

for task_name in tasks:
    for seed in tqdm(seeds):
        subprocess.run([
            "python", "sprint/sfc_seed_iter.py",
            "--task_name", task_name,
            "--seed", str(seed),
        ])
