import os
from sentence_transformers import util

def find_best_match(term, embeddings, model, top_k=5, log_dir=None, log_prefix=None):

    # Encode the term using the provided SentenceTransformer model
    term_emb = model.encode(term, normalize_embeddings=True)

    # Compute similarity scores with all embeddings in the Neo4j KG
    similarities = []
    for entity, embedding in embeddings.items():
        score = util.cos_sim(term_emb, embedding).item()
        similarities.append((entity, score))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Log the scores if a log directory is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{log_prefix or 'best_match'}_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Term: {term}\n")
            f.write("Top matches with similarity scores:\n")
            for entity, score in similarities[:top_k]:
                f.write(f"{entity}: {score:.4f}\n")
            f.write("\n")

    # Return the top-k matches
    return similarities[:top_k]