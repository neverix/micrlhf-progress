import os
from sprint.task_vector_utils import load_tasks

# Load tasks
tasks = load_tasks()


task_names = list(tasks.keys())

import subprocess

from tqdm.auto import tqdm

n_parts = 4

batched_task_names = [task_names[i::n_parts] for i in range(n_parts)]

from threading import Thread

def run_part(part, task_names):
    # os.environ['XLA_VISIBLE_DEVICES'] = str(part)

    # os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
    # os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
    # Different per process:
    # os.environ["TPU_VISIBLE_DEVICES"] = str(part) # "1", "2", "3"

    for task_name in tqdm(task_names):
        subprocess.run(["python", "sprint/circuit_ablation_script.py", task_name, str(part)])


threads = [Thread(target=run_part, args=(i, batched_task_names[i])) for i in range(n_parts)]

for thread in threads:
    thread.start()

for thread in threads:
    thread.join()