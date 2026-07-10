import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import parafac

img = Image.open("../data/my_image.jpg").convert("RGB")
X = np.array(img, dtype=np.float64)

# CP uses ONE rank (not one per mode) -- number of rank-1 terms summed together
rank = 100

weights, factors = parafac(X, rank=rank, init="random", random_state=0)
X_hat = tl.cp_to_tensor((weights, factors))
X_hat_clipped = np.clip(X_hat, 0, 255).astype(np.uint8)

error = tl.norm(X - X_hat) / tl.norm(X)
print(f"CP rank={rank}, relative error: {error:.4f}")

original_size = X.size
compressed_size = sum(f.size for f in factors) + len(weights)
print(f"Compression ratio: {original_size / compressed_size:.2f}x")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(X.astype(np.uint8)); axes[0].set_title("Original"); axes[0].axis("off")
axes[1].imshow(X_hat_clipped); axes[1].set_title(f"CP rank={rank}\nerror={error:.3f}"); axes[1].axis("off")
plt.tight_layout()
plt.savefig("../results/cp_result.png", dpi=120)
plt.show()