import dspy
import dsp
from dotenv import load_dotenv
import os

# Load environment variables from the .env file (if present)
load_dotenv()

import json
secret_value_0 = os.getenv("GROQ_KEY")
# temperature = 0.3
temperature = 0.0
llama_70b = dsp.modules.groq_client.GroqLM(secret_value_0,
                                          "llama3-70b-8192",
                                           temperature=temperature)
llama_8b = dsp.modules.groq_client.GroqLM(secret_value_0,
                                         "llama3-8b-8192",
                                          temperature=temperature)
# llama_70b = llama_8b

dspy.settings.configure(lm=llama_70b)


probe_layer = 16
threshold = 0.01

from datasets import load_dataset

dataset = load_dataset("kisate-team/generated-explanations", split="train")
over_threshold = dataset.filter(lambda x: max(x["scale_tuning"]["selfsims"][probe_layer]) > threshold)

class Rating(dspy.Signature):
    """Determine whether at least one of the generated texts is non-generic and aligned with the theme. Return True if at least one of them is aligned, False otherwise."""
    theme = dspy.InputField(desc="The target theme")
    texts = dspy.InputField(desc="The generated texts")

    rating = dspy.OutputField(desc="Answer whether at least one of the texts is aligned with the theme. Should be either True or False.", prefix="The answer is:")

predictor = dspy.Predict(Rating)

save_path = "ratings_top3.jsonl"

if not os.path.exists(save_path):
    with open(save_path, "w") as f:
        pass

with open(save_path) as f:
    ratings = f.readlines()
    ratings = [json.loads(x) for x in ratings]
    ratings = {x["feature"]: x["rating"] for x in ratings}

for r in over_threshold:
    if r["feature"] in ratings:
        continue

    theme = r["explanation"]
    texts = r["generations"]["texts"]

    selfsims = r["scale_tuning"]["selfsims"][probe_layer]
    ces = r["scale_tuning"]["crossents"][0]

    selfsims_normalized = [(x - min(selfsims)) / (max(selfsims) - min(selfsims)) for x in selfsims]
    ces_normalized = [(x - min(ces)) / (max(ces) - min(ces)) for x in ces]

    alpha = 0.4

    metric = [alpha * x - (1 - alpha) * y for x, y in zip(selfsims_normalized, ces_normalized)]

    texts = sorted(enumerate(texts), key=lambda x: metric[int(x[0] / len(texts) * len(metric))], reverse=True)
    texts = [x[1] for x in texts[:3]]

    texts = "\n".join(
        f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
    )

    try:

        response = predictor(
            theme=theme,
            texts=texts
        )

        rating = response.rating

        with open(save_path, "a") as f:
            f.write(json.dumps(
                {
                    "feature": r["feature"],
                    "rating": rating
                }
            ) + "\n")
    except Exception as e:
        print(e)
        print(r)
        print("\n\n\n")

# explanation = explanation.partition("This neuron is looking for")[2].strip().strip(".").split("\n")[0].strip()