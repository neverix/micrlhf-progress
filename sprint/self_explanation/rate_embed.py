import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import torch
torch.set_grad_enabled(False)
import os

import json

# with open("results.jsonl") as f:
#     results = [json.loads(line) for line in f]

# results = [result for result in results if "scale" in result]
# results = [result for result in results if result["scale"] > 0]

model_path = 'Alibaba-NLP/gte-large-en-v1.5'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = outputs.last_hidden_state[:, 0]
 
# # (Optionally) normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())

def find_embeddings(texts):
    with torch.inference_mode():
        batch_dict = tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')

        outputs = model(**batch_dict)
        embeddings = outputs.last_hidden_state[:, 0]

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

# with open("gen_explanations_new.jsonl") as f:
with open("../../data/gen_explanations_ss.jsonl") as f:
    explanations = [json.loads(line) for line in f]

from datasets import load_dataset

dataset = load_dataset("kisate-team/feature-explanations", split="train")


r = explanations[100]

theme = dataset[r["id"]]["explanation"]
# theme = Theme(theme=theme)
texts = [x for x in r["explanation"]]
# texts = "\n".join(
#     f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
# )

# print(texts)
# print(theme)

# class Rating(dspy.Signature):
#     """Rate alignment of generated texts with a theme. 1 is generally on-topic, 0 is completely off-topic."""
#     theme = dspy.InputField(desc="The theme of the generated texts")
#     texts = dspy.InputField(desc="The generated texts")

#     rating = dspy.OutputField(desc="The rating of the generated texts. Should be either 1 or 0.", prefix="The overal rating for all texts is:")

# predictor = dspy.Predict(Rating)

# response = predictor(
#     theme=theme,
#     texts=texts
# )

def find_score(theme, texts):
    embed_0, *embeds = find_embeddings([theme] + texts)
    scores = (torch.stack(embeds) @ embed_0) * 100
    score = scores.mean().item()
    return score

print(find_score(theme, texts))

# explanation = FeatureExplainer().rate_texts(texts=texts, theme=theme)

# print(
#     response
# )
from tqdm import tqdm
all_ratings = []
for r in (bar := tqdm(explanations)):
    theme = dataset[r["id"]]["explanation"]
    texts = [x for x in r["explanation"]]
    # texts = "\n".join(
    #     f"{i+1}. \"{x}\"" for i, x in enumerate(texts)
    # )

    # response = predictor(
    #     theme=theme,
    #     texts=texts
    # )

    # rating = response.rating
    rating = find_score(theme, texts)
    all_ratings.append(rating)
    bar.set_postfix(rating=rating, avg_rating=sum(all_ratings) / len(all_ratings))

    with open("ratings_ss.jsonl", "a") as f:
        f.write(json.dumps(
            {
                "id": r["id"],
                "rating": rating
            }
        ) + "\n")

# explanation = explanation.partition("This neuron is looking for")[2].strip().strip(".").split("\n")[0].strip()