import subprocess

tasks = [
    
 'en_es',
    'location_continent',
#  'football_player_position',
 'location_religion',
 'location_language',
#  'person_profession',
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
 'algo_second'][4:]

from tqdm.auto import tqdm

for task in tqdm(tasks):
    subprocess.run(["python", "sprint/run_ie_correlation.py", "--task", task])