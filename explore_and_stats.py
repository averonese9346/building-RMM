#checking out specifics from each set of data (training, validating, testing)
#want to see number of entries, story length, and checking to see if there are any empty stories
#to be cleaned later

import json
import numpy as np
import re

#openign all 3 data sets to check them out and retrieve specific info
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("valid.json", "r", encoding="utf-8") as f:
    valid_data = json.load(f)

with open("test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

#printing the set size for each file (number of prompt-story pairs)
print("Train set size (number of prompt-story pairs):", len(train_data))
print("Valid set size (number of prompt-story pairs):", len(valid_data))
print("Test set size (number of prompt-story pairs):", len(test_data))

prompt_lengths = [len(d["prompt"].split()) for d in train_data]
story_lengths = [len(d["story"].split()) for d in train_data]

#printing stats for each story
print("Average prompt length in characters:", sum(prompt_lengths)/len(prompt_lengths))
print("Average story length in characters:", sum(story_lengths)/len(story_lengths))
print("Max story length in characters:", max(story_lengths))

empty_stories = [d for d in train_data if len(d["story"].strip()) == 0]
print("Number of empty stories:", len(empty_stories))


prompt_words = [len(d["prompt"].split()) for d in train_data]
story_words = [len(d["story"].split()) for d in train_data]

print("Average prompt word count:", np.mean(prompt_words))
print("Average story word count:", np.mean(story_words))

story_sentences = [len(re.split(r'[.!?]', d["story"])) for d in train_data]
print("Average sentences per story:", np.mean(story_sentences))