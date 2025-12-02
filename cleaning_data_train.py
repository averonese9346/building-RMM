#goal: to clean the data from the training set from kaggle. making sure there are no duplicate
#entries, no null, etc.

import json
import re
import statistics
import random

#loading the data for the test set using json
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

#these are all of the cleaning functions to ensure the data is properly cleaned.
def clean_text(s):
    if s is None:
        return ""

    #removing whitespace
    s = s.strip()

    #normalizing quotes and making sure quotes are being read properly
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("‘", "'").replace("’", "'")

    #normalizing the em dash and making sure the em dash is read properly
    s = s.replace("—", "-").replace("–", "-")

    #replacing <newline> tokens with real newlines to ensure each line is formatted correctly
    s = s.replace("<newline>", "\n")

    #collapsing repeated spaces/tabs (preserves real newlines)
    s = re.sub(r"[ \t]+", " ", s)

    return s.strip() #stripping again after cleaning (there may be additional white spaces after cleaning)

#wordcount function
def word_count(s):
    return len(s.split())

#applying all filtering and cleaning functions from above
cleaned = []
min_story_words = 30  #threshold to remove unusable short stories (must have at least 30 characters)

for d in train_data:
    prompt = clean_text(d["prompt"])
    story = clean_text(d["story"])

    #dropping empty or extremely short stories (we do not want empty stories or anything that isn't at least 30 characters)
    if len(story) == 0:
        continue
    if word_count(story) < min_story_words:
        continue

    cleaned.append({
        "prompt": prompt,
        "story": story
    })

print(f"Original dataset size: {len(train_data)}")
print(f"Cleaned dataset size:  {len(cleaned)}")

#gathering stats for cleaned data
prompt_lengths = [word_count(x["prompt"]) for x in cleaned]
story_lengths = [word_count(x["story"]) for x in cleaned]

print("\n--- Dataset Statistics ---")
print("Prompt length   → mean:", statistics.mean(prompt_lengths),
      "median:", statistics.median(prompt_lengths))
print("Story length    → mean:", statistics.mean(story_lengths),
      "median:", statistics.median(story_lengths))
print("Story min/max   →", min(story_lengths), "/", max(story_lengths))

#making sure there are no duplicates
pairs = set()
dupes = 0

for x in cleaned:
    pair = (x["prompt"], x["story"])
    if pair in pairs:
        dupes += 1
    else:
        pairs.add(pair)

print("Exact duplicate prompt/story pairs:", dupes)


#and finally saving the cleaned data set as .json file
with open("train_cleaned.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

print("\nSaved cleaned dataset → train_cleaned.json")
