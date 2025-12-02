#this code will convert each word into a token (integer).
#this also combines each prompt and story into one sequence
#the sequences have been padded to ensure they are the max_len (to match the 95 percentile)

import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#loading filtered fantasy datasets (the cleaned ones)
with open("train_fantasy.json", "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open("valid_fantasy.json", "r", encoding="utf-8") as f:
    valid_data = json.load(f)

with open("test_fantasy.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

#combining all text from prompts and stories into one training set to build the tokenizer
all_texts = [d["prompt"] + " " + d["story"] for d in train_data]

#initalizing the tokenizer
tokenizer = Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(all_texts)

#saving the vocabulary for future use
vocab = tokenizer.word_index
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Vocabulary size: {len(vocab)} words")

#converting text to sequences
def texts_to_sequences(dataset):
    sequences = []
    for d in dataset:
        text = d["prompt"] + " " + d["story"]
        seq = tokenizer.texts_to_sequences([text])[0]
        sequences.append(seq)
    return sequences

train_sequences = texts_to_sequences(train_data)
valid_sequences = texts_to_sequences(valid_data)
test_sequences = texts_to_sequences(test_data)

#determining max sequence length
import numpy as np
lengths = [len(seq) for seq in train_sequences]
max_len = int(np.percentile(lengths, 95))
print(f"Padding/truncating sequences to max length: {max_len}")

#padding sequences
train_padded = pad_sequences(train_sequences, maxlen=max_len, padding="post", truncating="post")
valid_padded = pad_sequences(valid_sequences, maxlen=max_len, padding="post", truncating="post")
test_padded = pad_sequences(test_sequences, maxlen=max_len, padding="post", truncating="post")

#and finally saving all .npy files
np.save("train_sequences.npy", train_padded)
np.save("valid_sequences.npy", valid_padded)
np.save("test_sequences.npy", test_padded)

print("Sequences saved! Ready for RNN training.")

