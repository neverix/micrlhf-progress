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

from pydantic import BaseModel, Field


class Activation(BaseModel):
    context: str = Field(description="The context for the neuron's activation, ending with the token it activates on")
    after: str = Field(description="Tokens generated after the target token")
    # context: str = Field(description="The context for the neuron's activation, ending with the token it activates on")
    activation: float = Field(description="The neuron's activation")
    # activation: float = Field(description="The neuron's activation")


class FeatureExplainer(dspy.FunctionalModule):
# class NeuronExplainer(dspy.FunctionalModule):
    def __init__(self, use_explanation=True):
        super().__init__()
        self.use_explanation = use_explanation

    @dspy.cot
    def explain_feature(self, feature_activations: list[Activation]) -> str:
    # def explain_neuron(self, neuron_activations: list[Activation]) -> str:
        '''We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.'''
        # '''We're studying features in a neural network. Each feature looks for some particular thing in a short document. Look at the parts of the document the feature activates for and summarize in a single sentence what the feature is looking for. Don't list examples of words.'''
        pass

    @dspy.predictor
    def predict_feature_activation(self,
    # def predict_neuron_activation(self,
                                   feature_activations: list[Activation],
                                #    neuron_activations: list[Activation],
                                   explanation: str,
                                   text: str) -> float:
        # """Predict neuron activation for a text based on an explanation. Output only the number corresponding to the activation and nothing else."""
        """Predict feature activation for a text based on an explanation. Output only the number corresponding to the activation and nothing else."""
        pass

    def forward(self,
                explain_activations: list[Activation],
                predict_texts: list[str]) -> list[float]:
        if self.use_explanation:
            explanation = self.explain_feature(feature_activations=explain_activations)
            # explanation = self.explain_neuron(neuron_activations=explain_activations)
            # print("Explanation:", explanation)
        else:
            explanation = ""

        predictions = []
        for text in predict_texts:
            prediction = self.predict_feature_activation(feature_activations=explain_activations,
            # prediction = self.predict_neuron_activation(neuron_activations=explain_activations,
                explanation=explanation,
                text=text
            )
            predictions.append(prediction)

        return dspy.Prediction(predicted_activations=predictions)


from transformers import AutoTokenizer
TOKENIZER = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

import pyarrow.parquet as pq

token_table = pq.read_table("tokensv4_hermes.parquet")

layers = [4, 8, 10, 12, 13, 14, 16, 18]

dfs = {
    layer: pq.read_table(f"weights/caches/phi-l{layer}-r4-st0.25x128-activationsv4_hermes.parquet").to_pandas() for layer in layers
}

def make_table(feature, layer, token_range, after_next):
  df = dfs[layer]

  table_feat = df[df["feature"] == feature].copy()
  table_feat = table_feat[table_feat["freq"] > 0].copy()
  table_feat["before"] = table_feat["token"].apply(
      lambda x: tokenizer.decode(token_table[max(0, x - token_range):x+1]["tokens"].to_numpy()))
  table_feat["after"] = table_feat["token"].apply(
      lambda x: "" if not after_next else (tokenizer.decode(token_table[x+1:x+1+after_next]["tokens"].to_numpy()))
  )

  table_feat = table_feat.sort_values(by=["activation"], ascending=False)
  return table_feat 

def explain_feature(feature, layer, n_samples=20, token_range=8, after_next=3):
    feature_table = make_table(feature, layer, token_range, after_next)
    tf = feature_table.head(n=min(n_samples, len(feature_table)))
    activs = list(map(tuple, tf[["before", "after", "activation"]].itertuples(index=False)))

    # print(activs)

    explain_activations=[Activation(context=u, after=v, activation=w) for u, v, w in activs]
    explanation = FeatureExplainer().explain_feature(feature_activations=explain_activations)
    explanation = explanation.partition("This neuron is looking for")[2].strip().strip(".").split("\n")[0].strip()

    return explanation

from tqdm.auto import tqdm
import json


with open("features_ext.json") as f:
  dataset = json.load(f)

dataset = [x for x in dataset if "explanation" not in x][784:]


explanations = []

with open("features_ext.json") as f:
  dataset = json.load(f)

dataset = [x for x in dataset if "explanation" in x]


explanations = []

for item in tqdm(dataset):
  layer = item["layer"]
  feature = item["feature"]
  explanation = explain_feature(feature, layer, n_samples=15, token_range=10)

  explanations.append(
      {"layer": layer, "version": 4, "feature": feature, "type": item["type"], "explanation": explanation}
  )

with open("explanations2.json", "w") as f:
    json.dump(explanations, f)
