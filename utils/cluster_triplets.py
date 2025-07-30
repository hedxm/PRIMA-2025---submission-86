
import os
import json
from typing import List, Dict, Optional 

from sklearn.cluster import KMeans 
from sentence_transformers import SentenceTransformer 
from tqdm import tqdm 

def cluster_triplets_util(
    triplets: List[Dict], 
    max_clusters: int = 7, 
    output_dir: Optional[str] = "outputs/inductive_reasoner/clusters", 
    embedding_model_name: str = 'all-MiniLM-L6-v2' 
) -> List[List[Dict]]: 
    """
    Clusters triplets into groups based on semantic similarity using sentence embeddings.
    Saves each input cluster to a file in the output_dir.
    """
    if not triplets:
        print("⚠️ No triplets provided for clustering.")
        return []

    if output_dir: # Only create if a path is given
        os.makedirs(output_dir, exist_ok=True)

    # Flatten triplets into string form for embedding
    texts = [f"{t.get('subject','')} {t.get('predicate','')} {t.get('object','')}" for t in triplets]

    print(f"🧪 Vectorizing {len(texts)} triplets with SentenceTransformer model: {embedding_model_name}...")
    try:
        model = SentenceTransformer(embedding_model_name)
    except Exception as e:
        print(f"🚨 ERROR: Could not load SentenceTransformer model '{embedding_model_name}': {e}")
        print("Returning all triplets as a single cluster as a fallback.")
        if output_dir and triplets:
            cluster_file_path = os.path.join(output_dir, "cluster_1_input.json")
            try:
                with open(cluster_file_path, "w", encoding="utf-8") as f:
                    json.dump(triplets, f, indent=2, ensure_ascii=False)
            except Exception as e_save: print(f"Error saving single cluster fallback: {e_save}")
        return [triplets] if triplets else []


    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    num_samples = embeddings.shape[0]
    actual_num_clusters = min(max_clusters, num_samples)
    
    if actual_num_clusters < 1: # Should not happen if triplets is not empty
        print("ℹ️ No samples for clustering after embedding. Returning empty.")
        return []
    if actual_num_clusters == 1 and num_samples > 0: # Only one cluster possible or requested
        print(f"ℹ️ Only 1 cluster will be formed for {num_samples} triplet(s).")
        if output_dir and triplets:
            cluster_file_path = os.path.join(output_dir, "cluster_1_input.json")
            try:
                with open(cluster_file_path, "w", encoding="utf-8") as f:
                    json.dump(triplets, f, indent=2, ensure_ascii=False)
            except Exception as e_save: print(f"Error saving single cluster: {e_save}")
        return [triplets] if triplets else []
    # If num_samples < actual_num_clusters (e.g. trying to make 5 clusters from 3 samples), 
    # KMeans will error. So, adjust actual_num_clusters.
    if num_samples < actual_num_clusters:
        print(f"⚠️ Warning: Number of samples ({num_samples}) is less than max_clusters ({max_clusters}). Adjusting to {num_samples} clusters.")
        actual_num_clusters = num_samples


    print(f"🤖 Clustering into {actual_num_clusters} groups using K-Means on embeddings...")
    try:
        kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init='auto') # n_init='auto' is new default
        labels = kmeans.fit_predict(embeddings)
    except Exception as e_kmeans:
        print(f"🚨 ERROR during K-Means clustering: {e_kmeans}")
        print("Returning all triplets as a single cluster as a fallback.")
        if output_dir and triplets:
            cluster_file_path = os.path.join(output_dir, "cluster_1_input.json")
            try:
                with open(cluster_file_path, "w", encoding="utf-8") as f:
                    json.dump(triplets, f, indent=2, ensure_ascii=False)
            except Exception as e_save: print(f"Error saving single cluster fallback on K-Means error: {e_save}")
        return [triplets] if triplets else []


    clusters = [[] for _ in range(actual_num_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(triplets[i])

    if output_dir:
        print(f"💾 Saving input clustered triplets to: {output_dir}")
        for i, cluster in enumerate(clusters):
            if cluster: # Only save if cluster is not empty
                cluster_file_path = os.path.join(output_dir, f"cluster_{i+1}_input.json")
                try:
                    with open(cluster_file_path, "w", encoding="utf-8") as f:
                        json.dump(cluster, f, indent=2, ensure_ascii=False)
                except Exception as e_save_loop: print(f"Error saving cluster {i+1}: {e_save_loop}")
    
    return [c for c in clusters if c] # Return only non-empty clusters
