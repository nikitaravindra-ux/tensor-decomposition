import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker

# --- Load image as a 3rd-order tensor: (Height, Width, Channels) ---
img = Image.open("../data/my_image.jpg").convert("RGB")
X = np.array(img, dtype=np.float64)
print("Tensor shape:", X.shape)

# --- Choose multilinear rank (r1, r2, r3) ---
rank = (100, 100, 3)

# --- Compute Tucker decomposition via HOSVD ---
core, factors = tucker(X, rank=rank)
print("Core shape:", core.shape)
for i, f in enumerate(factors):
    print(f"Factor {i} shape:", f.shape)

# --- Reconstruct the approximation ---
X_hat = tl.tucker_to_tensor((core, factors))
X_hat_clipped = np.clip(X_hat, 0, 255).astype(np.uint8)

# --- Measure error ---
error = tl.norm(X - X_hat) / tl.norm(X)
print(f"Relative reconstruction error: {error:.4f}")

# --- Compression ratio ---
original_size = X.size
compressed_size = core.size + sum(f.size for f in factors)
print(f"Compression ratio: {original_size / compressed_size:.2f}x")

# --- Visualize ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(X.astype(np.uint8)); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(X_hat_clipped); axes[1].set_title(f"Tucker rank={rank}\nerror={error:.3f}"); axes[1].axis("off")
plt.tight_layout()
plt.savefig("../results/tucker_result.png", dpi=120)
plt.show()