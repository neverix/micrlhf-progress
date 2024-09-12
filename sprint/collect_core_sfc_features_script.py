import subprocess

tasks = ["antonyms", "en_es", "es_en", "present_simple_gerund", "country_capital", "location_religion", "person_profession", "en_fr", "en_it", "fr_en", "algo_first", "algo_last"][-2:]


for task in tasks:
    subprocess.run(["python", "sprint/collect_core_sfc_features.py", "--task", task])