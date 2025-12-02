#creating buckets of 20% to split dataset into equal k parts
#training the model k times
#using 4 folds (80%) to train
#using 1 fold (20%) to validate
#rotate which is used for validation

#importing all packages
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

#loading data
data = np.load("train_sequences.npy")   # shape (N, seq_len)

print("Data loaded:", data.shape)

vocab_size = int(np.max(data)) + 1
seq_len = data.shape[1] - 1

print("Detected vocab size:", vocab_size)
print("Input sequence length:", seq_len)

#creating model
def build_model(vocab_size, seq_len):
    model = Sequential([
        Embedding(vocab_size, 256, input_length=seq_len),
        LSTM(256),
        Dense(vocab_size, activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam"
    )
    return model

#k-fold setup
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_losses = []

#now running each fold
fold = 1
for train_idx, val_idx in kfold.split(data):
    print(f"\n===== TRAINING FOLD {fold} =====")

    train_raw = data[train_idx]
    val_raw = data[val_idx]

    #Inputs = all but last token
    X_train = train_raw[:, :-1]
    X_val = val_raw[:, :-1]

    #Targets = last token only
    y_train = train_raw[:, -1]
    y_val = val_raw[:, -1]

    model = build_model(vocab_size, seq_len)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=16,
        verbose=1
    )

    val_loss = history.history["val_loss"][-1]
    fold_losses.append(val_loss)

    print(f"Fold {fold} final val_loss: {val_loss}")

    fold += 1

#printing the results and stats from each fold
print("\n===== CROSS-VALIDATION COMPLETE =====")
for i, loss in enumerate(fold_losses, start=1):
    print(f"Fold {i}: {loss}")

avg = sum(fold_losses) / len(fold_losses)
print("\nAverage validation loss:", avg)
