"""
Reproducibility script: regenerates data/products_vectorstore/ from scratch.

Ported from the "Now create a Chroma Datastore" cell in
llm_engineering/week8/day2.ipynb (the cell that creates the "products"
Chroma collection by loading Item.from_hub(...), batching through
SentenceTransformer.encode(), and calling collection.add(...)).

The vectorstore is normally just *copied* from the course repo
(cp -r week8/products_vectorstore data/products_vectorstore) rather
than built - this script exists purely so the store can be regenerated if it
is ever lost or needs to be rebuilt against a different dataset.

Requires packages not part of the standard backend runtime deps:
    uv sync --extra data

The selected ed-donner/items_full dataset is public and does not require a token.

Run from repo root:
    uv run python backend/scripts/build_vectorstore.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from app.agents.items import Item
load_dotenv(override=True)

# Build the product index from the full dataset supplied for this checkout.
LITE_MODE = False

HF_USERNAME = "ed-donner"
DATASET_NAME = f"{HF_USERNAME}/items_lite" if LITE_MODE else f"{HF_USERNAME}/items_full"

COLLECTION_NAME = "products"
BATCH_SIZE = 1000
PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTORSTORE_PATH = Path(
    os.environ.get("PRODUCTS_VECTORSTORE_PATH", PROJECT_ROOT / "data/products_vectorstore")
)


def main() -> None:
    print(f"Loading dataset {DATASET_NAME} ...")
    train, val, test = Item.from_hub(DATASET_NAME)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")

    client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    existing_collection_names = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing_collection_names:
        print(f"Collection '{COLLECTION_NAME}' already exists at {VECTORSTORE_PATH} - nothing to do.")
        return

    collection = client.create_collection(COLLECTION_NAME)
    for i in tqdm(range(0, len(train), BATCH_SIZE)):
        batch = train[i : i + BATCH_SIZE]
        documents = [item.summary for item in batch]
        vectors = encoder.encode(documents).astype(float).tolist()
        metadatas = [{"category": item.category, "price": item.price} for item in batch]
        ids = [f"doc_{j}" for j in range(i, i + len(documents))]
        collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    print(f"Built collection '{COLLECTION_NAME}' with {len(train):,} items at {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()
