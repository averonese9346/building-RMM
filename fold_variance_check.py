#compiling simple stats for each fold to check out the sequence length
import numpy as np

data = np.load("train_sequences.npy")
#compute simple stats for each fold (e.g. average sequence length before padding)
#(if you kept lengths; otherwise skip)
#Check token id distribution in problematic folds
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for i, (train_idx, val_idx) in enumerate(kfold.split(data), 1):
    val = data[val_idx]
    #compute mean non-pad tokens per sample (count > 0)
    mean_nonpad = np.mean((val > 0).sum(axis=1))
    print(f"Fold {i} mean non-pad tokens:", mean_nonpad)
