import os
import json
import math
import numpy as np
from tools.triplet_matcher import match_against_neo4j 

def run_triplet_matcher_batches(
    triplet_file_to_match,
    kg_triplet_embeddings_path, 
    kg_triplet_texts_path, 
    threshold=0.8,
    batch_size=25,
    output_dir="outputs/triplet_matcher_results" 
):

    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(triplet_file_to_match, "r", encoding="utf-8") as f:
            input_triplets = json.load(f)
    except FileNotFoundError:
        print(f"🚨 ERROR: Input triplet file for matching not found: {triplet_file_to_match}")
        # Create empty result files so downstream doesn't break if it expects these files
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump([], f_u)
        return # Cannot proceed
    except json.JSONDecodeError:
        print(f"🚨 ERROR: Could not decode JSON from input triplet file: {triplet_file_to_match}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump([], f_u)
        return

    if not input_triplets:
        print(f"ℹ️ No input triplets to match from {triplet_file_to_match}. Saving empty results.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump([], f_u)
        return

    # Preprocess input triplets to ensure they have S-P-O and are clean
    preprocessed_input_triplets = []
    for triplet in input_triplets:
        if isinstance(triplet, dict) and all(key in triplet for key in ["subject", "predicate", "object"]):
            preprocessed_input_triplets.append({
                "subject": str(triplet["subject"]).strip(),
                "predicate": str(triplet["predicate"]).strip(),
                "object": str(triplet["object"]).strip()
            })
        else:
            print(f"⚠️ WARNING: Skipping invalid input triplet during preprocessing for matcher: {triplet}")
    
    if not preprocessed_input_triplets:
        print(f"ℹ️ No valid input triplets after preprocessing from {triplet_file_to_match}. Saving empty results.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump([], f_u)
        return

    print(f"DEBUG: Preprocessed input triplets for matching: {preprocessed_input_triplets[:3]}")

    # Load KG embeddings and texts (these are for the *entire* KG, precomputed)
    try:
        print(f"Triplet Matcher: Loading KG triplet embeddings from: {kg_triplet_embeddings_path}")
        kg_embeddings = np.load(kg_triplet_embeddings_path)
        print(f"Triplet Matcher: Loading KG triplet texts from: {kg_triplet_texts_path}")
        with open(kg_triplet_texts_path, "r", encoding="utf-8") as f:
            kg_texts = json.load(f)
    except FileNotFoundError as e:
        print(f"🚨 ERROR: KG embedding files for matching not found: {e}")
        print("Ensure KG triplet embeddings and texts are precomputed and paths are correct.")
        # If KG data is missing, all input triplets are "unmatched"
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump(preprocessed_input_triplets, f_u, indent=2)
        return
    except Exception as e:
        print(f"🚨 ERROR loading KG embedding data: {e}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump(preprocessed_input_triplets, f_u, indent=2)
        return

    if kg_embeddings.size == 0 or not kg_texts:
        print("⚠️ KG embeddings or texts are empty. All input triplets will be considered unmatched.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f_m: json.dump([], f_m)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f_u: json.dump(preprocessed_input_triplets, f_u, indent=2)
        return


    total = len(preprocessed_input_triplets)
    num_batches = math.ceil(total / batch_size)
    
    all_matched_results = []
    all_unmatched_results = []
    log_lines = []

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, total)
        batch_to_match = preprocessed_input_triplets[start:end]
        print(f"\n🔎 Matching batch {i + 1}/{num_batches} ({len(batch_to_match)} triplets)")

        batch_match_results = match_against_neo4j(
            triplets_to_check=batch_to_match, 
            kg_embeddings=kg_embeddings, 
            kg_texts=kg_texts, 
            threshold=threshold
        )

        current_batch_all_processed = batch_match_results.get("matched_triplets", []) + \
                                      batch_match_results.get("unmatched_triplets", [])
        
        for r_item in current_batch_all_processed:
            line = f"{'✅' if r_item['match'] else '❌'} {r_item['subject']} -- {r_item['predicate']} --> {r_item['object']} | Score: {r_item.get('similarity', 0.0):.4f}"
            log_lines.append(line)
            print(line)

        all_matched_results.extend(batch_match_results.get("matched_triplets", []))
        all_unmatched_results.extend(batch_match_results.get("unmatched_triplets", []))

    # Save results
    with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f:
        json.dump(all_matched_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f:
        json.dump(all_unmatched_results, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "matcher_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"\n✅ Matched (vs KG full triplets): {len(all_matched_results)}")
    print(f"❌ Unmatched (vs KG full triplets): {len(all_unmatched_results)}")
    print(f"📁 Matcher results saved to: {output_dir}")
