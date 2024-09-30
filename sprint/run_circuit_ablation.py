from sprint.task_vector_utils import load_tasks

# Load tasks
tasks = load_tasks()


task_names = list(tasks.keys())

import subprocess

from tqdm.auto import tqdm


for task_name in tqdm(task_names):
    print(f"Running {task_name}")
    subprocess.run(["python", "sprint/circuit_ablation_script.py", task_name])
    print(f"Finished {task_name}")
