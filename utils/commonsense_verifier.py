import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from utils.flat_triplets import flat_triplets_util

def load_numberbatch_embeddings(emb_path):
    print("🔍 Loading Numberbatch embeddings...")
    concept2vec = {}
    with open(emb_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            concept = parts[0].replace('_', ' ').lower()
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            concept2vec[concept] = vector
    print(f"✅ Loaded {len(concept2vec)} concepts from Numberbatch\n")
    return concept2vec

def verify_commonsense_hybrid(
    triplet_file,
    emb_path,
    threshold_nb=0.2,
    threshold_e5=0.8,
    output_dir="outputs/commonsense_verifier",
    verified_file_name="verified_triplets.json",
    rejected_file_name="rejected_triplets.json",
    log_file_name="verification_log.txt"
):
    os.makedirs(output_dir, exist_ok=True)

    with open(triplet_file, "r", encoding="utf-8") as f:
        triplets_raw = json.load(f)
        
    triplets_raw = flat_triplets_util(triplets_raw)
    triplets = []
    for t in triplets_raw:
        subj = t.get("subject", "").strip()
        pred = t.get("predicate", "").strip()
        obj = t.get("object", "").strip()
        if subj and pred and obj:
            triplets.append((subj, pred, obj))

    concept2vec = load_numberbatch_embeddings(emb_path)
    e5 = SentenceTransformer("intfloat/e5-small-v2")

    accepted = []
    rejected = []
    log_lines = []

    print(f"🔎 Verifying {len(triplets)} triplets using hybrid strategy...\n")
    for h, r, t in triplets:
        h_norm = h.lower().replace('_', ' ')
        t_norm = t.lower().replace('_', ' ')
        r_norm = r.lower().replace('_', ' ')

        hv_nb = concept2vec.get(h_norm)
        tv_nb = concept2vec.get(t_norm)

        nb_score = None
        e5_score = None
        predicate_score = None
        used = "Numberbatch"
        fallback = False

        # Calculate Numberbatch score if embeddings are available
        if hv_nb is not None and tv_nb is not None:
            nb_score = cosine_similarity(hv_nb.reshape(1, -1), tv_nb.reshape(1, -1))[0][0]
            nb_score = (nb_score + 1) / 2  # Normalize to [0, 1]
        else:
            nb_score = 0.4  # Default score when embeddings are missing

        # Richer query using predicate semantics
        query = f"{h} is related to {t} via '{r}'"
        doc = f"{t}"  # You can also use t here if you want
        query_emb = e5.encode(query, normalize_embeddings=True)
        doc_emb = e5.encode(doc, normalize_embeddings=True)
        e5_score = util.cos_sim(query_emb, doc_emb).item()
        e5_score = max(0, min(e5_score, 1))  # Ensure score is in [0, 1]

        # Compute predicate-to-entity similarity
        predicate_query = f"{h} {r} {t}"
        predicate_emb = e5.encode(predicate_query, normalize_embeddings=True)
        entity_emb = e5.encode(f"{h} {t}", normalize_embeddings=True)
        predicate_score = util.cos_sim(predicate_emb, entity_emb).item()
        predicate_score = max(0, min(predicate_score, 1))  # Ensure score is in [0, 1]

        # Determine final score
        if hv_nb is not None and tv_nb is not None:
            final_score = (0.5 * e5_score + 0.3 * nb_score + 0.2 * predicate_score)
            threshold = 0.8
        else:
            final_score = (0.6 * e5_score + 0.4 * predicate_score)
            threshold = 0.7

        # Log individual scores for debugging
        print(f"DEBUG: h={h}, r={r}, t={t}, nb_score={nb_score}, e5_score={e5_score}, predicate_score={predicate_score}, final_score={final_score}")

        # Determine acceptance based on stricter combined score
        accepted_flag = final_score >= threshold
        flag = "✅" if accepted_flag else "❌"
        log_line = f"{flag} {h:20} -- {r:12} --> {t:20} | NB: {nb_score:.4f}, E5: {e5_score:.4f}, Pred: {predicate_score:.4f}, Final: {final_score:.4f} ({used})"
        print(log_line)
        log_lines.append(log_line)

        # Add to accepted or rejected lists
        triplet_obj = {"subject": h, "predicate": r, "object": t, "score": final_score, "method": used}
        if accepted_flag:
            accepted.append(triplet_obj)
        else:
            rejected.append(triplet_obj)

    # Save outputs
    with open(os.path.join(output_dir, verified_file_name), "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, rejected_file_name), "w", encoding="utf-8") as f:
        json.dump(rejected, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, log_file_name), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n✅ Verified: {len(accepted)} | ❌ Rejected: {len(rejected)}")
    print(f"📁 Saved results in: {output_dir}")