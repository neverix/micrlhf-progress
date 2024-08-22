import subprocess

tasks = ["antonyms", "en_es", "es_en", "present_simple_gerund"]


for task in tasks:
    subprocess.run(["python", "sprint/run_icl_sfc.py", "--task", task])