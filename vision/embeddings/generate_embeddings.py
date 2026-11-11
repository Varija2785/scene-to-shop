from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, AutoProcessor


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

OBJECTS_DIR = (
    BASE_DIR
    / "vision"
    / "detection"
    / "outputs"
    / "objects_clean"
)

VERIFICATION_FILE = (
    BASE_DIR
    / "vision"
    / "verification"
    / "outputs"
    / "verified_objects.json"
)

OUTPUT_DIR = (
    BASE_DIR
    / "vision"
    / "embeddings"
    / "outputs"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

MODEL_NAME = "openai/clip-vit-base-patch32"

print("=" * 70)
print("SCENE-TO-SHOP | VISUAL EMBEDDING GENERATION")
print("=" * 70)

print()
print(f"Loading model: {MODEL_NAME}")
print("The first run may download the model weights.")
print()

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)

model.eval()

print("CLIP model loaded.")
print()


# ---------------------------------------------------------
# LOAD VERIFICATION RESULTS
# ---------------------------------------------------------

with open(VERIFICATION_FILE, "r", encoding="utf-8") as f:
    verified_objects = json.load(f)


# Only keep objects Gemini considers shopping relevant.
shopping_objects = [
    item
    for item in verified_objects
    if item.get("shopping_relevant") is True
]

print(
    f"Shopping-relevant objects found: "
    f"{len(shopping_objects)}"
)

print()


# ---------------------------------------------------------
# GENERATE EMBEDDINGS
# ---------------------------------------------------------

results = []

for index, item in enumerate(shopping_objects, start=1):

    filename = item["source_image"]

    image_path = next(
        OBJECTS_DIR.rglob(filename),
        None
    )

    if image_path is None:
        print(f"[{index}] Image not found: {filename}")
        continue

    print(
        f"[{index}/{len(shopping_objects)}] "
        f"{item['object_name']}"
    )

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # -----------------------------------------------------
    # CLIP IMAGE FEATURES
    # -----------------------------------------------------

    with torch.inference_mode():
        image_features = model.get_image_features(
            **inputs
        )

    # Some Transformers versions return an object
    # containing pooler_output instead of a tensor.
    if hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output

    # Convert tensor to NumPy.
    embedding = (
        image_features
        .detach()
        .cpu()
        .numpy()[0]
    )

    # -----------------------------------------------------
    # L2 NORMALIZATION
    # -----------------------------------------------------

    embedding = embedding / np.linalg.norm(embedding)

    # -----------------------------------------------------
    # STORE RESULT
    # -----------------------------------------------------

    result = {
        "source_image": filename,
        "object_name": item["object_name"],
        "object_category": item["object_category"],
        "description": item["description"],
        "attributes": item["attributes"],
        "verification_confidence": item["confidence"],
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(len(embedding)),
        "embedding": embedding.tolist()
    }

    results.append(result)

    print(
        f"  Embedding dimension: {len(embedding)}"
    )

    print()


# ---------------------------------------------------------
# SAVE EMBEDDINGS
# ---------------------------------------------------------

output_file = (
    OUTPUT_DIR
    / "object_embeddings.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2
    )


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("=" * 70)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 70)

print(
    f"Objects embedded: {len(results)}"
)

if results:
    print(
        f"Embedding dimension: "
        f"{results[0]['embedding_dimension']}"
    )

print()
print("Saved to:")
print(output_file)