import dspy
import dsp
import os
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

import json

# with open("results.jsonl") as f:
#     results = [json.loads(line) for line in f]

# results = [result for result in results if "scale" in result]
# results = [result for result in results if result["scale"] > 0]


with open("gen_explanations_new.jsonl") as f:
    explanations = [json.loads(line) for line in f]

from datasets import load_dataset

dataset = load_dataset("kisate-team/feature-explanations", split="train")


r = explanations[100]

theme = dataset[r["id"]]["explanation"]
# theme = Theme(theme=theme)
texts = [x for x in r["explanation"]]
texts = "\n".join(
    f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
)

# print(texts)
# print(theme)

class Rating(dspy.Signature):
    """Rate alignment of generated texts with a theme. 1 is generally on-topic, 0 is completely off-topic."""
    theme = dspy.InputField(desc="The theme of the generated texts")
    texts = dspy.InputField(desc="The generated texts")

    rating = dspy.OutputField(desc="The rating of the generated texts. Should be either 1 or 0.", prefix="The overal rating for all texts is:")

predictor = dspy.Predict(Rating)

response = predictor(
    theme=theme,
    texts=texts
)

# explanation = FeatureExplainer().rate_texts(texts=texts, theme=theme)

# print(
#     response
# )

for r in explanations:
    theme = dataset[r["id"]]["explanation"]
    texts = [x for x in r["explanation"]]
    texts = "\n".join(
        f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
    )

    response = predictor(
        theme=theme,
        texts=texts
    )

    rating = response.rating

    with open("ratings_new.jsonl", "a") as f:
        f.write(json.dumps(
            {
                "id": r["id"],
                "rating": rating
            }
        ) + "\n")

# explanation = explanation.partition("This neuron is looking for")[2].strip().strip(".").split("\n")[0].strip()