import os
from sprint.task_vector_utils import load_tasks

# Load tasks
tasks = load_tasks()


task_names = list(tasks.keys())[:12]

import subprocess

from tqdm.auto import tqdm

n_parts = 4

batched_task_names = [
    task_names[i * n_parts : (i + 1) * n_parts] for i in range(len(task_names) // n_parts + 1)
]


for batch in tqdm(batched_task_names):
    subprocess.run(
        [
            "python",
            "sprint/circuit_ablation_script_2.py",
            ",".join(batch),
        ]
    )