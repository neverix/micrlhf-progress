import subprocess

_tasks = [
 'en_es',
'location_continent',
 'football_player_position',
 'location_religion',
 'location_language',
 'person_profession',
 'location_country',
 'country_capital',
 'person_language',
 'singular_plural',
 'present_simple_past_simple',
 'antonyms',
 'plural_singular',
 'present_simple_past_perfect',
 'present_simple_gerund',
 'en_it',
 'it_en',
 'en_fr',
 'fr_en',
 'es_en',
 'algo_last',
 'algo_first',
 'algo_second']

tasks = [
    "antonyms",
    "en_es",
    "person_profession",
    "location_country",
    "es_en",
    "singular_plural",
    "en_fr",
    "present_simple_past_simple",
]

tasks = tasks + [x for x in _tasks if x not in tasks]


import subprocess

from tqdm.auto import tqdm

n_parts = 4

batched_task_names = [
    tasks[i * n_parts : (i + 1) * n_parts] for i in range(len(tasks) // n_parts + 1)
]


for batch in tqdm(batched_task_names):
    subprocess.run(["python", "sprint/run_ie_correlation.py", ",".join(batch)])