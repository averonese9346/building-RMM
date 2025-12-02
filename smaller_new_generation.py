#run this code to generate my fantasy text (character arc and backstory)

import sentencepiece as spm
import tensorflow as tf
import numpy as np
import textwrap

#loading my model and trained tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("fantasy_tokenizer.model")

model = tf.keras.models.load_model("fantasy_model_fast.h5")
print("Model and tokenizer loaded successfully.\n")

EOS_ID = tokenizer.piece_to_id("</s>")

#the function to generate text
def generate_text(prompt, max_length=150, temperature=0.8):
    ids = tokenizer.encode(prompt, out_type=int)

    for _ in range(max_length):
        x = np.array([ids[-30:]])

        logits = model.predict(x, verbose=0)[0]
        logits = np.log(logits + 1e-9) / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))

        next_id = np.random.choice(len(probs), p=probs)

        if next_id < 0 or next_id >= tokenizer.get_piece_size():
            break

        if next_id == EOS_ID:
            break

        ids.append(next_id)

    #decoding the full output
    text = tokenizer.decode(ids)

#ensuring that a wrap is incldued at 200 characters to avoid text overflow on one line
    wrapped = textwrap.fill(text, width=200)

    return wrapped


#and running main function!
if __name__ == "__main__":
    prompt = "Create an adventurous backstory for a bard named Brecken who is an elf and a flirt."
    print("===== GENERATED TEXT =====\n")
    output = generate_text(prompt)
    print(output)
