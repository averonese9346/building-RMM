#code to analyze data files from kaggle
#goal: just want to make sure the data files from kaggle can be opened correctly and
#are reflecting correct information, analyzing the cleaned data

import json

#loading the training data set
with open("train.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

#checking out the first three entries
for entry in train_data[:3]:
    print("Prompt:", entry["prompt"])
    print("Story:", entry["story"][:100], "...")
    print("----")

#loading the validation data set
with open("valid.json", "r", encoding="utf-8") as f:
    valid_data = json.load(f)

#checking out the first three entries
for entry in valid_data[:3]:
    print("Prompt:", entry["prompt"])
    print("Story:", entry["story"][:100], "...")
    print("----")

#loading the test data set
with open("test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

#checking out the first three entries
for entry in test_data[:3]:
    print("Prompt:", entry["prompt"])
    print("Story:", entry["story"][:100], "...")
    print("----")