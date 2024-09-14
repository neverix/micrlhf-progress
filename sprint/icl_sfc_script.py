import subprocess

tasks = ["antonyms", "en_es", "es_en", "present_simple_gerund", "country_capital", "location_religion", "person_profession", "en_fr", "en_it", "fr_en"]


for task in tasks:
    subprocess.run(["python", "sprint/run_icl_sfc.py", "--task", task])