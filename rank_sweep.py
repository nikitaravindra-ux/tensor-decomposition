import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker, parafac

# --- Load image once ---
img = Image.open("data/my_image.jpg").convert("RGB")
X = np.array(img, dtype=np.float64)
original_size = X.size

# --- Ranks to test ---
tucker_ranks = [10, 30, 50, 75, 100, 150, 200]
cp_ranks     = [10, 30, 50, 75, 100, 150, 200]

tucker_errors, tucker_compressions = [], []
cp_errors, cp_compressions = [], []

# --- Tucker sweep ---
print("Running Tucker sweep...")
for r in tucker_ranks:
    rank = (r, r, 3)
    core, factors = tucker(X, rank=rank)
    X_hat = tl.tucker_to_tensor((core, factors))

    error = tl.norm(X - X_hat) / tl.norm(X)
    compressed_size = core.size + sum(f.size for f in factors)
    compression = original_size / compressed_size

    tucker_errors.append(error)
    tucker_compressions.append(compression)
    print(f"  r={r:4d} | error={error:.4f} | compression={compression:.2f}x")

# --- CP sweep ---
print("\nRunning CP sweep...")
for r in cp_ranks:
    weights, factors = parafac(X, rank=r, init="random", random_state=0)
    X_hat = tl.cp_to_tensor((weights, factors))

    error = tl.norm(X - X_hat) / tl.norm(X)
    compressed_size = sum(f.size for f in factors) + len(weights)
    compression = original_size / compressed_size

    cp_errors.append(error)
    cp_compressions.append(compression)
    print(f"  r={r:4d} | error={error:.4f} | compression={compression:.2f}x")

# --- Plot: the actual rate-distortion curve ---
plt.figure(figsize=(8, 6))
plt.plot(tucker_compressions, tucker_errors, "o-", label="Tucker", color="#4C72B0")
plt.plot(cp_compressions, cp_errors, "s-", label="CP", color="#DD8452")
plt.xlabel("Compression Ratio (x)")
plt.ylabel("Relative Reconstruction Error")
plt.title("Compression vs. Error: Tucker vs. CP")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("results/rank_sweep_comparison.png", dpi=120)
plt.show()