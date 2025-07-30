import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def precompute_embeddings(csv_path, emb_output_path, text_output_path, model_name="all-MiniLM-L6-v2"):
    os.makedirs(os.path.dirname(emb_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

    try:
        df = pd.read_csv(csv_path, delimiter="\t")
    except FileNotFoundError:
        print(f"🚨 ERROR: CSV file not found at {csv_path}. Cannot precompute triplet embeddings.")
        np.save(emb_output_path, np.array([])) # Create empty files
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return
    except Exception as e:
        print(f"🚨 ERROR: Could not read CSV {csv_path}: {e}")
        np.save(emb_output_path, np.array([])) # Create empty files
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return


    df.dropna(subset=["subject", "predicate", "object"], inplace=True)
    df = df[
        df["subject"].astype(str).str.strip().astype(bool) &
        df["predicate"].astype(str).str.strip().astype(bool) &
        df["object"].astype(str).str.strip().astype(bool)
    ]

    if df.empty:
        print(f"⚠️ No valid triplets found in {csv_path} after cleaning. Skipping triplet embedding generation.")
        np.save(emb_output_path, np.array([]))
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return

    triplet_texts = [
        f"{row['subject']} {row['predicate']} {row['object']}"
        for _, row in df.iterrows()
    ]

    print(f"DEBUG: Loaded {len(triplet_texts)} triplets for embedding: {triplet_texts[:3]}")

    print(f"🔍 Loading SBERT model: {model_name} for triplets")
    model = SentenceTransformer(model_name)

    print(f"💡 Computing embeddings for {len(triplet_texts)} triplets...")
    embeddings = model.encode(triplet_texts, show_progress_bar=True, normalize_embeddings=True)

    np.save(emb_output_path, embeddings)
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(triplet_texts, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved triplet embeddings: {emb_output_path}")
    print(f"📄 Saved triplet texts: {text_output_path}")


def precompute_single_column_embeddings(csv_path, column_name, emb_output_path, text_output_path, model_name="all-MiniLM-L6-v2"):
    """
    Precompute embeddings for items from a single column in a CSV file.
    Assumes comma-separated CSV by default unless specified otherwise.
    """
    os.makedirs(os.path.dirname(emb_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(text_output_path), exist_ok=True)

    try:
        df = pd.read_csv(csv_path) # Assumes comma delimiter for single column usually
    except FileNotFoundError:
        print(f"🚨 ERROR: CSV file not found at {csv_path}. Cannot precompute single-column embeddings.")
        np.save(emb_output_path, np.array([])) # Create empty files
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return
    except Exception as e:
        print(f"🚨 ERROR: Could not read CSV {csv_path}: {e}")
        np.save(emb_output_path, np.array([])) # Create empty files
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return

    if column_name not in df.columns:
        print(f"🚨 ERROR: Column '{column_name}' not found in {csv_path}. Available columns: {df.columns.tolist()}")
        np.save(emb_output_path, np.array([])) # Create empty files
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return

    df.dropna(subset=[column_name], inplace=True)
    df = df[df[column_name].astype(str).str.strip().astype(bool)]


    if df.empty:
        print(f"⚠️ No valid items found in column '{column_name}' from {csv_path} after cleaning. Skipping embedding generation.")
        np.save(emb_output_path, np.array([]))
        with open(text_output_path, "w", encoding="utf-8") as f: json.dump([], f)
        return

    texts = df[column_name].astype(str).tolist()
    print(f"DEBUG: Loaded {len(texts)} items for embedding from column '{column_name}': {texts[:3]}")

    print(f"🔍 Loading SBERT model: {model_name} for column '{column_name}'")
    model = SentenceTransformer(model_name)

    print(f"💡 Computing embeddings for {len(texts)} items from '{column_name}'...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    np.save(emb_output_path, embeddings)
    with open(text_output_path, "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved embeddings for '{column_name}' to: {emb_output_path}")
    print(f"📄 Saved texts for '{column_name}' to: {text_output_path}")
