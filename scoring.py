#gathering stats from my generated text to compare against the test set

import numpy as np
import tensorflow as tf
import sentencepiece as spm

#loading tokenizer and model
tokenizer = spm.SentencePieceProcessor(model_file="fantasy_tokenizer.model")
model = tf.keras.models.load_model("fantasy_model_fast.h5")

#main function for computing loss and perplexity to compare against test
def compute_loss_and_perplexity(text):
    ids = tokenizer.encode(text, out_type=int)

    #Shift to create input->label pairs
    x = np.array(ids[:-1])[None, :]
    y = np.array(ids[1:])[None, :]

    #Get logits
    logits = model.predict(x, verbose=0)

    #Compute cross entropy
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')

    loss_per_token = loss_fn(y, logits).numpy()
    avg_loss = loss_per_token.mean()
    ppl = np.exp(avg_loss)

    return avg_loss, ppl

#my generated text
paragraph = """<needed the line of the guard and my spinning are statues ... He 's now . `` Charlotte , I 've all ? '' `` It was at
a few inse existence , '' All joined the created 's be people anything opportunity and stop . You 's killed it 's a heavy to found a trench , it ' must a own complex . She 'd no at work . Theitchm 50
asks . `` Do I 's going the make . `` The whole right of the walked . And the step ! Heense , I have in the King . '' `` Well as my few bottle , the seat , I have the dead of the sisters and be men .
`` paradise 's life . I have not sort a gunfire friend . ''>"""

loss, ppl = compute_loss_and_perplexity(paragraph)
print("Loss:", loss)
print("Perplexity:", ppl)
