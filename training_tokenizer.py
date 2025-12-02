#training tokenizer to predict next word
#training on smaller data set, last one was too large

import sentencepiece as spm
import tensorflow as tf
import numpy as np
import os
import random

#loading the data
full_corpus_path = "fantasy_corpus.txt"
small_corpus_path = "fantasy_small.txt"

#creating a smaller corpus for faster training (last tokenizer would have taken approx. 1 week)
if not os.path.exists(small_corpus_path):
    with open(full_corpus_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sample = lines[:1000]  #taking first 1000 lines
    with open(small_corpus_path, "w", encoding="utf-8") as f:
        f.writelines(sample)

    print("Created fantasy_small.txt (1000 lines).")

#actually training the tokenizer
if not os.path.exists("fantasy_tokenizer.model"):
    print("Training tokenizer...")

    spm.SentencePieceTrainer.train(
        input=small_corpus_path,
        model_prefix="fantasy_tokenizer",
        vocab_size=5000,
        model_type="bpe",
        max_sentence_length=20000
    )

    print("Tokenizer trained successfully.")
else:
    print("Tokenizer already exists. Skipping.")

#loading tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("fantasy_tokenizer.model")

#handling special tokens
EOS_ID = tokenizer.piece_to_id("</s>")

#prepping training data
def encode_text(text):
    ids = tokenizer.encode(text, out_type=int)
    ids.append(EOS_ID)
    return ids

with open(small_corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

encoded = encode_text(raw_text)

#building sequences for training
sequence_length = 30  #needed sequences to be shorter to ensure an accurate measure of training time
inputs = []
targets = []

for i in range(len(encoded) - sequence_length):
    seq_in = encoded[i:i+sequence_length]
    seq_out = encoded[i+sequence_length]
    inputs.append(seq_in)
    targets.append(seq_out)

inputs = np.array(inputs)
targets = np.array(targets)

print("Prepared training data:", inputs.shape)

#building the model
vocab_size = tokenizer.get_piece_size()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128),  # reduced for speed
    tf.keras.layers.GRU(256, return_sequences=False),
    tf.keras.layers.Dense(vocab_size, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy"
)

model.summary()

#training with 3 epochs to ensure faster training
EPOCHS = 3
BATCH_SIZE = 64

history = model.fit(
    inputs,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

#and finally saving the model
model.save("fantasy_model_fast.h5")
print("Model saved as fantasy_model_fast.h5")
