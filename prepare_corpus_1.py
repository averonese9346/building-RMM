#the corpus is the file that the model will be trained on (combining all sets)
#like combining all samples so that we can just load this file
#into training and begin generation

import json

datasets = ["train_fantasy.json", "valid_fantasy.json", "test_fantasy.json"]
corpus_path = "fantasy_corpus.txt"

all_texts = []

for ds in datasets:
    with open(ds, "r", encoding="utf-8") as f:
        data = json.load(f)
    for sample in data:
        text = sample.get("story", "")
        if text.strip():
            all_texts.append(text.strip())

with open(corpus_path, "w", encoding="utf-8") as f:
    for t in all_texts:
        f.write(t + "\n")

print("Corpus prepared with", len(all_texts), "samples")
