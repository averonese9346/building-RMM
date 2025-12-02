#testing scores from the test set
#gathering stats from test set to evaluate the potential for the model to succeed

import numpy as np
import json
import math
import tensorflow as tf

#loadign sequences and vocab
data_test = np.load("test_sequences.npy")
#inputs are all but last token, targets are last token
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

model = tf.keras.models.load_model("fantasy_text_lstm_model.h5")
test_loss = model.evaluate(X_test, y_test, batch_size=64)
test_perplexity = math.exp(test_loss)

print("Test loss:", test_loss)
print("Test perplexity:", test_perplexity)
