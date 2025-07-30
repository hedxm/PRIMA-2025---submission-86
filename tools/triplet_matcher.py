import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os # For model caching path if needed

# It's good practice to initialize the model once if possible,
# but within this function scope is also fine if it's not too frequent.
# For heavy use, consider passing the model as an argument.
MODEL_NAME = "all-MiniLM-L6-v2"
# You can set a cache folder for sentence-transformers
# os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/path/to/cache_folder'

try:
    # Attempt to load model once when module is loaded for efficiency in some scenarios
    # This might not be ideal if the function is part of a larger app with complex loading
    # print(f"DEBUG: Pre-loading SentenceTransformer model: {MODEL_NAME} in triplet_matcher.py")
    # _internal_sbert_model = SentenceTransformer(MODEL_NAME)
    _internal_sbert_model = None # Or initialize it here
except Exception as e:
    print(f"Warning: Could not pre-load SentenceTransformer model: {e}")
    _internal_sbert_model = None


def get_sbert_model(model_name=MODEL_NAME):
    global _internal_sbert_model
    if _internal_sbert_model is None:
        print(f"DEBUG: Initializing SentenceTransformer model: {model_name} in get_sbert_model")
        _internal_sbert_model = SentenceTransformer(model_name)
        print("DEBUG: SentenceTransformer model initialized successfully.")
    return _internal_sbert_model


def match_against_neo4j(triplets_to_check, kg_embeddings, kg_texts, threshold=0.8):
    """
    Compare a batch of triplets against precomputed Neo4j KG embeddings.

    Args:
        triplets_to_check (list of dict): Triplets to check (S-P-O format).
        kg_embeddings (numpy.ndarray): Precomputed embeddings of the KG texts.
        kg_texts (list of str): The KG texts corresponding to kg_embeddings.
        threshold (float): Cosine similarity threshold for match.

    Returns:
        dict: Contains "matched_triplets" and "unmatched_triplets".
    """
    # print(f"DEBUG: Received triplets for matching: {triplets_to_check[:3]}") # Debug
    # print(f"DEBUG: KG embeddings shape: {kg_embeddings.shape if kg_embeddings is not None else 'None'}") # Debug
    # print(f"DEBUG: KG texts count: {len(kg_texts) if kg_texts is not None else 'None'}") # Debug


    if not triplets_to_check:
        return {"matched_triplets": [], "unmatched_triplets": []}
    
    if kg_embeddings is None or kg_embeddings.size == 0 or not kg_texts:
        print("⚠️ match_against_neo4j: KG embeddings or texts are empty. All input triplets considered unmatched.")
        # Add match:false and similarity:0 to all input triplets
        unmatched_with_details = []
        for t in triplets_to_check:
            unmatched_with_details.append({
                "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
                "match": False, "similarity": 0.0, "matched_kg_text": None
            })
        return {"matched_triplets": [], "unmatched_triplets": unmatched_with_details}

    model = get_sbert_model() # Use shared or cached model instance

    matched_triplets_results = []
    unmatched_triplets_results = []

    # Prepare texts from input triplets for batch encoding
    input_texts_to_embed = []
    valid_input_triplets_for_embedding = []
    for triplet in triplets_to_check:
        if not (isinstance(triplet, dict) and all(key in triplet for key in ["subject", "predicate", "object"])):
            print(f"ERROR: Invalid triplet format in match_against_neo4j: {triplet}")
            # Still add it to unmatched with default values if needed, or just skip
            unmatched_triplets_results.append({**triplet, "match": False, "similarity": 0.0, "matched_kg_text": None}) # Add original if possible
            continue
        
        s = str(triplet["subject"]).strip()
        p = str(triplet["predicate"]).strip()
        o = str(triplet["object"]).strip()
        
        if not (s and p and o): # Ensure S, P, O are not empty after stripping
             print(f"ERROR: Empty S, P, or O in triplet: {triplet}")
             unmatched_triplets_results.append({
                "subject": s, "predicate": p, "object": o,
                "match": False, "similarity": 0.0, "matched_kg_text": None
            })
             continue

        input_texts_to_embed.append(f"{s} {p} {o}")
        valid_input_triplets_for_embedding.append(triplet) # Keep original triplet for associating results

    if not input_texts_to_embed: # All input triplets were invalid
        return {"matched_triplets": [], "unmatched_triplets": unmatched_triplets_results}

    # Batch encode the input triplet texts
    # print(f"DEBUG: Encoding {len(input_texts_to_embed)} input triplet texts for matching...") # Debug
    query_vecs = model.encode(input_texts_to_embed, normalize_embeddings=True, show_progress_bar=False) # Batch encoding

    # Perform cosine similarity for all query_vecs against all kg_embeddings at once
    # sims_matrix will have shape (len(query_vecs), len(kg_embeddings))
    sims_matrix = cosine_similarity(query_vecs, kg_embeddings)

    for i, original_triplet in enumerate(valid_input_triplets_for_embedding):
        s = str(original_triplet["subject"]).strip()
        p = str(original_triplet["predicate"]).strip()
        o = str(original_triplet["object"]).strip()
        
        sims_for_this_query = sims_matrix[i]
        best_idx = int(np.argmax(sims_for_this_query))
        best_score = float(sims_for_this_query[best_idx])

        is_match = best_score >= threshold
        result_details = {
            "subject": s,
            "predicate": p,
            "object": o,
            "match": is_match,
            "similarity": best_score,
            "matched_kg_text": kg_texts[best_idx] if kg_texts and best_idx < len(kg_texts) else None # Safety for kg_texts
        }

        if is_match:
            matched_triplets_results.append(result_details)
        else:
            unmatched_triplets_results.append(result_details)
            
    # If there were initially invalid triplets, they are already in unmatched_triplets_results without S,P,O
    # This might lead to mixed content. Better to ensure all items in unmatched_triplets_results have S,P,O.
    # The current logic for invalid triplets already adds them with S,P,O if available.

    return {"matched_triplets": matched_triplets_results, "unmatched_triplets": unmatched_triplets_results}
