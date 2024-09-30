import subprocess

from sprint.task_vector_utils import load_tasks, ICLRunner
tasks = load_tasks()

task_names = [
    "en_es",
    "country_capital",
    "person_language",
    "es_en",
    "en_fr",
    "algo_last",
    "plural_singular"
]

task_names = set(tasks.keys()) - set(task_names)

task_names = ["location_continent"]

# task_names = list(tasks.keys())

# task_names = [
#     "antonyms"
# ]

from tqdm.auto import tqdm

for task_name in tqdm(task_names):
    subprocess.run(["python", "sprint/run_icl_sfc_edge_ies.py", "--task", task_name])