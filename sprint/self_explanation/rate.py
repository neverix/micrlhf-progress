import dspy
import dsp
secret_value_0 = ""
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

from pydantic import BaseModel, Field


class RatingRequest(BaseModel):
    theme: str = Field(description="The theme of the generated texts")
    texts: str = Field(description="The generated texts")

class FeatureExplainer(dspy.FunctionalModule):
# class NeuronExplainer(dspy.FunctionalModule):
    def __init__(self, use_explanation=True):
        super().__init__()
        self.use_explanation = use_explanation

    @dspy.cot
    def rate_texts(self, theme:str, texts: str) -> str:
    # def explain_neuron(self, neuron_activations: list[Activation]) -> str:
        '''We had a model generate several texts with a single particular theme. Look at the generated texts and provide a rating of how well the model captured the theme. Rate the model's performance on a scale of 1 to 5, where 1 is the texts being completely off-topic and 5 is the texts being perfectly on-topic. Give a single rating for all texts. Output the rating first and then any additional comments.'''
        pass


import json


with open("gen_explanations.jsonl") as f:
    explanations = [json.loads(line) for line in f]

from datasets import load_dataset

dataset = load_dataset("kisate-team/feature-explanations", split="train")


r = explanations[100]

theme = dataset[r["id"]]["explanation"]
texts = [x for x in r["explanation"]]
texts = "\n".join(
    f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
)


class Rating(dspy.Signature):
    """Rate alignment of generated texts with a theme. 1 is completely off-topic, 5 is generally on-topic."""
    theme = dspy.InputField(desc="The theme of the generated texts")
    texts = dspy.InputField(desc="The generated texts")

    rating = dspy.OutputField(desc="The rating of the generated texts. Should be a value between 1 and 5.", prefix="The overal rating for all texts is:")

predictor = dspy.Predict(Rating)

response = predictor(
    theme=theme,
    texts=texts
)


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

    with open("ratings.jsonl", "a") as f:
        f.write(json.dumps(
            {
                "id": r["id"],
                "rating": rating
            }
        ) + "\n")