#creating the fold loss table and plot for project
import csv
import matplotlib.pyplot as plt
import math

fold_losses = [0.41501864790916443, 0.39288002252578735, 0.39912015199661255,
               0.6215232610702515, 0.5705291628837585]

#saving as a csv first
with open("kfold_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "val_loss"])
    for i, loss in enumerate(fold_losses, 1):
        writer.writerow([i, loss])
    writer.writerow(["mean", sum(fold_losses)/len(fold_losses)])

#now plotting
plt.figure(figsize=(6,4))
plt.plot(range(1, len(fold_losses)+1), fold_losses, marker='o')
plt.title("Validation Loss Across 5 Folds")
plt.xlabel("Fold")
plt.ylabel("Validation Loss (cross-entropy)")
plt.grid(True)
plt.savefig("kfold_val_loss.png", dpi=150)
plt.show()

#printing mean and perplexities
avg_loss = sum(fold_losses)/len(fold_losses)
print("Average loss:", avg_loss)
print("Average perplexity:", math.exp(avg_loss))
for i,l in enumerate(fold_losses,1):
    print(f"Fold {i} perplexity:", math.exp(l))
