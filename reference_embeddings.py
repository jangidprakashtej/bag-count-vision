"""
Build reference embeddings for each product from your sample images.

Run once (and again whenever you add/change product images):
    python reference_embeddings.py

Instead of one blurry "average" embedding per product, this uses K-means
to split each product's sample images into a few sub-clusters (e.g. one
for "front view", one for "side view") and saves each cluster centroid
as its own prototype. This gives better matching accuracy when your
sample photos vary in angle/lighting, since a live crop only needs to be
close to ONE of a product's prototypes, not a single averaged blend of
all of them.

Produces product_embeddings.json:
    {
      "product_a": [[vector], [vector], ...],   # one vector per K-means cluster
      "product_b": [[vector], [vector], ...],
      ...
    }
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"

# Max clusters (prototypes) to create per product. Actual number used per
# product is min(MAX_PROTOTYPES_PER_PRODUCT, number of sample images for
# that product) so this is safe even for products with very few photos.
MAX_PROTOTYPES_PER_PRODUCT = 3


def load_model():
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return model, processor


def embed_image(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.squeeze(0).cpu().tolist()


def main():
    model, processor = load_model()
    embeddings = {}

    for product in config.PRODUCT_LIST:
        folder = os.path.join(config.REFERENCE_IMAGES_DIR, product)
        if not os.path.isdir(folder):
            print(f"[WARN] no folder found for '{product}' at {folder}, skipping")
            continue

        vectors = []
        for fname in os.listdir(folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, fname)
            try:
                vectors.append(embed_image(model, processor, path))
            except Exception as e:
                print(f"[WARN] failed on {path}: {e}")

        if not vectors:
            print(f"[WARN] no usable images for '{product}'")
            continue

        vectors_arr = np.array(vectors)  # (num_images, embedding_dim)

        # k can't exceed the number of images we have for this product
        k = min(MAX_PROTOTYPES_PER_PRODUCT, len(vectors_arr))

        if k == 1:
            # only one usable image (or MAX_PROTOTYPES_PER_PRODUCT=1) -> no
            # real clustering possible, just use it directly as the prototype
            prototypes = vectors_arr.tolist()
        else:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(vectors_arr)
            centroids = kmeans.cluster_centers_
            # re-normalize centroids (K-means averages vectors, which can
            # shrink their length; cosine similarity later assumes unit length)
            centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
            prototypes = centroids.tolist()

        embeddings[product] = prototypes
        print(f"[OK] {product}: {len(vectors)} images -> {len(prototypes)} prototype(s) (k={k})")

    with open(config.EMBEDDINGS_FILE, "w") as f:
        json.dump(embeddings, f)
    print(f"\nSaved {len(embeddings)} product embeddings to {config.EMBEDDINGS_FILE}")


if __name__ == "__main__":
    main()
