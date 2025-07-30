import os
import json
import math
from pathlib import Path 
from typing import Callable, List, Dict, Any
import traceback 


try:
    from utils.flat_triplets import flat_triplets_util
except ImportError:
    def flat_triplets_util(data): return data if isinstance(data, list) else []


try:
    import numpy as np
    from tools.triplet_matcher import match_against_neo4j
except ImportError:
    print("Warning: numpy or tools.triplet_matcher not found. run_triplet_matcher_batches might fail if called.")
    def match_against_neo4j(*args, **kwargs): return {"matched_triplets": [], "unmatched_triplets": []}


def run_agent_in_batches(
    agent_fn: Callable[..., Dict[str, Any]],
    agent_name: str,
    input_data: List[Any],
    batch_key: str,
    output_key: str = None,
    output_dir: str = None,
    logs_dir: str = None,
    batch_size: int = 25,
    verbose: bool = True,
    **agent_fn_kwargs: Any
) -> List[Any]:

    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True) 

    total_items = len(input_data)
    if total_items == 0 and verbose:
        print(f"ℹ️ Agent '{agent_name}': No input data to process.")
        return []
        
    # Calculate num_batches, ensuring at least 1 batch if there's data
    num_batches = math.ceil(total_items / batch_size) if total_items > 0 else 0
    all_agent_outputs = [] # Stores the direct dict output from each agent_fn call

    if verbose: print(f"🚀 Running agent '{agent_name}' in {num_batches} batches (batch size: {batch_size})...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_items)
        current_batch_data = input_data[start_idx:end_idx]

        if not current_batch_data: continue 
        
        if verbose: print(f"\nProcessing batch {i + 1}/{num_batches} ({len(current_batch_data)} items) for '{agent_name}'...")

        batch_input_arg = {batch_key: current_batch_data} if batch_key else current_batch_data
        current_agent_fn_output = {} # Default to empty dict on error

        try:
            current_agent_fn_output = agent_fn(
                batch_input_arg,
                log_dir=logs_dir,
                batch_idx=i,
                **agent_fn_kwargs
            )
            if not isinstance(current_agent_fn_output, dict):
                 print(f"🚨 WARNING: agent_fn for '{agent_name}' batch {i+1} did not return a dict (got {type(current_agent_fn_output)}). Using empty dict.")
                 current_agent_fn_output = {} # Replace with empty dict

            all_agent_outputs.append(current_agent_fn_output)

            if output_dir: 
                 data_to_save_for_batch = current_agent_fn_output

                 batch_output_filename = os.path.join(output_dir, f"{agent_name}_batch_{i+1}_output.json")
                 try:
                     with open(batch_output_filename, "w", encoding="utf-8") as f_batch:
                         json.dump(data_to_save_for_batch, f_batch, indent=2, ensure_ascii=False)
                 except Exception as e_batch_save:
                     if verbose: print(f"🚨 WARNING: Could not save output for batch {i+1}: {e_batch_save}")
            # --- END: ADDED PER-BATCH SAVING --
            
        except Exception as e:
            print(f"🚨 ERROR processing batch {i + 1} for agent '{agent_name}': {e}")
            traceback.print_exc()
            all_agent_outputs.append({}) 
            if output_dir:
                error_filename = os.path.join(output_dir, f"{agent_name}_batch_{i+1}_error.txt")
                with open(error_filename, "w", encoding="utf-8") as f_err:
                    f_err.write(f"Error processing batch {i+1}:\n{e}\n")
                    f_err.write(traceback.format_exc())

    final_results_to_return = []
    if output_key:
        if verbose: print(f"\nExtracting key '{output_key}' from batch results...")
        for i, batch_output_dict in enumerate(all_agent_outputs):
            if isinstance(batch_output_dict, dict):
                extracted_part = batch_output_dict.get(output_key)
                if extracted_part is not None:
                    final_results_to_return.append(extracted_part)
                else:
                    if verbose: print(f"  ⚠️ Key '{output_key}' not found or None in result for batch {i+1}.")
                    final_results_to_return.append([]) # Append empty list as default for missing key
            else:
                if verbose: print(f"  ⚠️ Expected dict but got {type(batch_output_dict)} for batch {i+1}. Appending empty list.")
                final_results_to_return.append([])
    else:
        # No output_key specified, return the list of full output dictionaries from agent_fn
        if verbose: print("\nNo output_key specified, returning full batch results.")
        final_results_to_return = all_agent_outputs

    # --- Save FINAL COMBINED/Processed results ---
    if output_dir:
        
        data_to_save_combined = []
        if output_key: 
             combined_items = [item for sublist in final_results_to_return if isinstance(sublist, list) for item in sublist]
             data_to_save_combined = combined_items
        else: 
            data_to_save_combined = final_results_to_return

        combined_filename = f"{agent_name}_all_batches_combined_output.json"
        if output_key:
             combined_filename = f"{agent_name}_all_batches_combined_{output_key}.json"
             
        final_output_path = os.path.join(output_dir, combined_filename)
        try:
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save_combined, f, indent=2, ensure_ascii=False)
            if verbose: print(f"✅ Saved combined results for '{agent_name}' to: {final_output_path}")
        except TypeError as te:
            if verbose: print(f"🚨 ERROR: Could not serialize combined results for '{agent_name}' to JSON: {te}.")
        except Exception as e:
            if verbose: print(f"🚨 ERROR: Could not save combined results for '{agent_name}': {e}")

    if verbose:
        count_desc = "raw agent output items" if not output_key else f"items extracted via '{output_key}'"
        print(f"✅ Agent '{agent_name}' processing completed. Returning {len(final_results_to_return)} {count_desc}.")
    
    return final_results_to_return


def run_triplet_matcher_batches(
    triplet_file_to_match,
    kg_triplet_embeddings_path,
    kg_triplet_texts_path,
    threshold=0.8,
    batch_size=25,
    output_dir="outputs/triplet_matcher_results"
):

    if 'np' not in globals(): import numpy as np # Import only if needed

    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(triplet_file_to_match, "r", encoding="utf-8") as f:
            input_triplets = json.load(f)
    except FileNotFoundError:
        print(f"🚨 ERROR (Matcher): Input triplet file not found: {triplet_file_to_match}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump([], fu)
        return
    except json.JSONDecodeError:
        print(f"🚨 ERROR (Matcher): Could not decode JSON from: {triplet_file_to_match}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump([], fu)
        return

    if not input_triplets:
        print(f"ℹ️ (Matcher): No input triplets to match from {triplet_file_to_match}.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump([], fu)
        return

    preprocessed_input_triplets = []
    for triplet in input_triplets:
        if isinstance(triplet, dict) and all(key in triplet for key in ["subject", "predicate", "object"]):
            preprocessed_input_triplets.append({
                "subject": str(triplet["subject"]).strip(),
                "predicate": str(triplet["predicate"]).strip(),
                "object": str(triplet["object"]).strip()
            })

    if not preprocessed_input_triplets:
        print(f"ℹ️ (Matcher): No valid triplets after preprocessing from {triplet_file_to_match}.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump([], fu)
        return

    try:
        kg_embeddings = np.load(kg_triplet_embeddings_path)
        with open(kg_triplet_texts_path, "r", encoding="utf-8") as f: kg_texts = json.load(f)
    except FileNotFoundError as e:
        print(f"🚨 ERROR (Matcher): KG embedding files not found: {e}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump(preprocessed_input_triplets, fu, indent=2)
        return
    except Exception as e:
        print(f"🚨 ERROR (Matcher): loading KG embedding data: {e}")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump(preprocessed_input_triplets, fu, indent=2)
        return

    if kg_embeddings.size == 0 or not kg_texts:
        print("⚠️ (Matcher): KG embeddings/texts empty. All input triplets considered unmatched.")
        with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as fm: json.dump([], fm)
        with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as fu: json.dump(preprocessed_input_triplets, fu, indent=2)
        return

    total = len(preprocessed_input_triplets)
    num_batches = math.ceil(total / batch_size)
    all_matched_results, all_unmatched_results, log_lines = [], [], []

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, total)
        batch_to_match = preprocessed_input_triplets[start:end]
        if not batch_to_match: continue

        batch_match_results = match_against_neo4j(
            triplets_to_check=batch_to_match, kg_embeddings=kg_embeddings, kg_texts=kg_texts, threshold=threshold
        )

        current_batch_all_processed = batch_match_results.get("matched_triplets", []) + batch_match_results.get("unmatched_triplets", [])
        for r_item in current_batch_all_processed:
            line = f"{'✅' if r_item.get('match') else '❌'} {r_item.get('subject','')} -- {r_item.get('predicate','')} --> {r_item.get('object','')} | Score: {r_item.get('similarity', 0.0):.4f}"
            log_lines.append(line)

        all_matched_results.extend(batch_match_results.get("matched_triplets", []))
        all_unmatched_results.extend(batch_match_results.get("unmatched_triplets", []))

    with open(os.path.join(output_dir, "matched_triplets.json"), "w", encoding="utf-8") as f: json.dump(all_matched_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "unmatched_triplets.json"), "w", encoding="utf-8") as f: json.dump(all_unmatched_results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "matcher_log.txt"), "w", encoding="utf-8") as f: f.write("\n".join(log_lines))

    print(f"\n✅ Matcher: Matched={len(all_matched_results)}, Unmatched={len(all_unmatched_results)}")
    print(f"📁 Matcher results saved to: {output_dir}")
