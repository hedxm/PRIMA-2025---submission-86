import json
import os
import re
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import traceback

from utils.flat_triplets import flat_triplets_util
from utils.find_best_match import find_best_match
from utils import run_agent_in_batches

try:
    from tools import clean_text_for_json
except ImportError:
    print("ERROR: clean_text_for_json not found.")
    def clean_text_for_json(text): return text


def escape_cypher_string(value: str) -> str:
    if not isinstance(value, str): value = str(value)
    return value.replace('\\', '\\\\').replace("'", "\\'")

def format_relationship_type(predicate: str) -> str:
    if not predicate: return "RELATED_TO"
    s = re.sub(r'[^\w\s_]', '', predicate); s = re.sub(r'\s+', '_', s)
    s = s.upper(); s = s.strip('_')
    return s if s else "RELATED_TO"

def kg_curator_node(config):
    labeler_llm_model_name = config.get("llm", "llama3:instruct")
    labeler_llm = ChatOllama(model=labeler_llm_model_name)
    sbert_model_for_matching = SentenceTransformer(config["embedding_model"])

    try:
        with open(config["labeler_prompt_path"], "r", encoding="utf-8") as f:
            labeler_prompt_template_str = f.read()
    except Exception as e:
        print(f"🚨 ERROR: Loading labeler prompt {config['labeler_prompt_path']}: {e}.")
        labeler_prompt_template_str = "ERROR: PROMPT NOT LOADED"

    labeler_prompt = PromptTemplate(
        template=labeler_prompt_template_str,
        input_variables=["entity_list_json", "target_labels_list_str", "default_label"],
        template_format="jinja2"
    )

    target_labels = config.get("target_kg_labels", [])
    default_label_name = config.get("default_label", "Review")
    if default_label_name not in target_labels:
        target_labels.append(default_label_name)
    target_labels_str = ", ".join([f"`{label}`" for label in target_labels])

    MAX_RETRIES_LABELER = config.get("max_labeler_retries", 3)


    def agent_fn(batch_input_dict: Dict[str, Any],
                 log_dir: Optional[str] = None,
                 batch_idx: Optional[int] = None,
                 **kwargs) -> Dict[str, Any]:

      
        globally_unique_mapped_triplets_for_batch = batch_input_dict.get("triplets", [])
        if not globally_unique_mapped_triplets_for_batch:
            return {"triplets": [], "queries": [], "labeled_entities_map": {}}

        entities_to_label_for_this_batch: Set[str] = set()
        for t in globally_unique_mapped_triplets_for_batch:
            if t.get("subject"): entities_to_label_for_this_batch.add(t["subject"])
            if t.get("object"): entities_to_label_for_this_batch.add(t["object"])

        llm_assigned_label_map_for_batch: Dict[str, str] = {}
        entity_list_to_classify_for_batch = sorted(list(entities_to_label_for_this_batch))

        if entity_list_to_classify_for_batch:
            print(f"INFO: Calling Labeler LLM for {len(entity_list_to_classify_for_batch)} unique entities (KG Processing Batch {batch_idx}, LLM Call 1)...")

            llm_failed = False
            for attempt in range(MAX_RETRIES_LABELER):
                try:
                    entity_list_json_str = json.dumps(entity_list_to_classify_for_batch)
                    filled_prompt = labeler_prompt.format(
                        entity_list_json=entity_list_json_str,
                        target_labels_list_str=target_labels_str,
                        default_label=default_label_name
                    )
                    response = labeler_llm.invoke(filled_prompt)
                    output_text = response.content if hasattr(response, "content") else str(response)

                    if log_dir and batch_idx is not None:
                        raw_path = os.path.join(log_dir, f"labeler_raw_output_batch_{batch_idx+1}_attempt_{attempt+1}.txt")
                        try:
                            with open(raw_path, "w", encoding="utf-8") as f_log:
                                f_log.write(output_text) # Log only LLM output
                        except Exception as e_log:
                            print(f"⚠️ Error writing labeler log: {e_log}")
                    
                    cleaned_output_text = clean_text_for_json(output_text)
                    if not cleaned_output_text:
                        raise ValueError("Cleaned LLM output for labeler empty.")

                    parsed_map = json.loads(cleaned_output_text)
                    if not isinstance(parsed_map, dict):
                        raise ValueError(f"LLM labeler output not dict: {type(parsed_map)}")

                    temp_map = {}
                    invalid_labels_found = []
                    llm_output_normalized_keys = {key.lower().strip(): key for key in parsed_map.keys()}

                    for entity_original_case in entity_list_to_classify_for_batch:
                        entity_normalized = entity_original_case.lower().strip()
                        label_to_assign = default_label_name

                        if entity_normalized in llm_output_normalized_keys:
                            original_llm_key = llm_output_normalized_keys[entity_normalized]
                            label_from_llm = str(parsed_map[original_llm_key]).strip()
                            if label_from_llm in target_labels:
                                label_to_assign = label_from_llm
                            else:
                                invalid_labels_found.append(label_from_llm)
                        temp_map[entity_original_case] = label_to_assign
                    
                    llm_assigned_label_map_for_batch = temp_map
                    
                    missing_entities = [e for e in entity_list_to_classify_for_batch if e not in llm_assigned_label_map_for_batch] # Check against original list passed to LLM
                    if missing_entities:
                        print(f"⚠️ LLM labeler did not return labels for all requested entities in its map. Assigning default to: {missing_entities} (Batch {batch_idx})")
                        for entity in missing_entities:
                            if entity not in llm_assigned_label_map_for_batch: # Safeguard
                                llm_assigned_label_map_for_batch[entity] = default_label_name
                    
                    if invalid_labels_found:
                        print(f"⚠️ Labeler LLM invalid labels: {set(invalid_labels_found)}. Used default '{default_label_name}'. Batch: {batch_idx}")
                    
                    print(f"✅ Labeler LLM Success (Attempt {attempt+1}, Batch {batch_idx}).")
                    llm_failed = False
                    break
                except Exception as e_label:
                    print(f"🚨 ERROR in Labeler LLM call/parsing (Batch {batch_idx}, Attempt {attempt+1}): {e_label}")
                    traceback.print_exc()
                    llm_failed = True
                    if attempt == MAX_RETRIES_LABELER - 1:
                        print(f"❌ All retries failed for labeler LLM. Assigning default label '{default_label_name}' for entities in Batch {batch_idx}.")
                        llm_assigned_label_map_for_batch = {entity: default_label_name for entity in entity_list_to_classify_for_batch}
                        break
        else:
            print(f"INFO: No unique entities to label in Batch {batch_idx}.")

        generated_queries_for_batch: List[str] = []
        final_triplets_output_for_batch: List[Dict[str, Any]] = []

        for triplet in globally_unique_mapped_triplets_for_batch:
            try:
                subject_label = llm_assigned_label_map_for_batch.get(triplet["subject"], default_label_name)
                object_label = llm_assigned_label_map_for_batch.get(triplet["object"], default_label_name)

                final_triplets_output_for_batch.append({
                    **triplet,
                    "subject_label": subject_label,
                    "object_label": object_label
                })

                cypher_subject = escape_cypher_string(triplet["subject"])
                cypher_object = escape_cypher_string(triplet["object"])
                cypher_predicate = format_relationship_type(triplet["predicate"])

                if not (cypher_subject and cypher_object and cypher_predicate and subject_label and object_label):
                    print(f"⚠️ Skipping query generation for incomplete triplet data (Batch {batch_idx}): s='{triplet['subject']}', p='{triplet['predicate']}', o='{triplet['object']}' with labels s_lbl='{subject_label}', o_lbl='{object_label}'")
                    continue
                
                query = f"MERGE (s:{subject_label} {{name: '{cypher_subject}'}}) MERGE (o:{object_label} {{name: '{cypher_object}'}}) MERGE (s)-[:{cypher_predicate}]->(o)"
                generated_queries_for_batch.append(query)
            except Exception as e_query:
                print(f"🚨 Error generating Cypher for labeled triplet (Batch {batch_idx}) {triplet}: {e_query}")

        return {
            "triplets": final_triplets_output_for_batch, 
            "queries": generated_queries_for_batch,
        }


    def invoke(state: Dict[str, Any]) -> Dict[str, Any]:
        print("🚀 Entered KG Curator (Global Pre-deduplication, Batched Agent Version)")

        input_triplets_raw = state.get("unmatched_triplets", [])
        if not input_triplets_raw:
            return {"curated_triplets": [], "cypher_queries": []}
        print(f"📦 KG Curator processing {len(input_triplets_raw)} raw 'unmatched' triplets...")

        cleaned_input_triplets = [
            t for t in input_triplets_raw
            if isinstance(t, dict) and all(k in t for k in ["subject", "predicate", "object"])
        ]
        if not cleaned_input_triplets:
            print("ℹ️ No valid triplets after basic cleaning.")
            return {"curated_triplets": [], "cypher_queries": []}
        
        flattened_input_triplets = flat_triplets_util(cleaned_input_triplets)
        print(f"✅ Flattened to {len(flattened_input_triplets)} triplets for initial mapping and deduplication.")

        try:
            entity_embs_np = np.load(config["neo4j_entity_embeddings_path"])
            with open(config["neo4j_entity_text_path"], "r", encoding="utf-8") as f: entity_texts_list = json.load(f)
            if len(entity_texts_list) != entity_embs_np.shape[0]: raise ValueError("Entity text/embedding mismatch")
            neo4j_entity_embeddings_map = dict(zip(entity_texts_list, entity_embs_np))
            
            relation_embs_np = np.load(config["neo4j_relation_embeddings_path"])
            with open(config["neo4j_relation_text_path"], "r", encoding="utf-8") as f: relation_texts_list = json.load(f)
            if len(relation_texts_list) != relation_embs_np.shape[0]: raise ValueError("Relation text/embedding mismatch")
            neo4j_relation_embeddings_map = dict(zip(relation_texts_list, relation_embs_np))
            print(f"✅ KG Curator: Loaded Entity ({len(neo4j_entity_embeddings_map)}) and Relation ({len(neo4j_relation_embeddings_map)}) embeddings.")
        except Exception as e:
            print(f"🚨 ERROR: KG Curator loading embeddings: {e}")
            return {"curated_triplets": flattened_input_triplets, "cypher_queries": []}

        # --- Step 1: Perform find_best_match mapping for ALL triplets ---
        all_mapped_triplets_intermediate: List[Dict[str, str]] = []
        print(f"INFO: Starting global mapping for {len(flattened_input_triplets)} triplets...")
        for i, triplet in enumerate(flattened_input_triplets):
            subject = str(triplet.get("subject", "")).strip()
            predicate = str(triplet.get("predicate", "")).strip()
            obj = str(triplet.get("object", "")).strip()
            if not (subject and predicate and obj): continue

            try:
                similar_subjects = find_best_match(term=subject, embeddings=neo4j_entity_embeddings_map, model=sbert_model_for_matching, top_k=1, log_dir=config.get("matcher_log_dir"), log_prefix=f"curator_global_subj_{i}")
                mapped_subject = similar_subjects[0][0] if similar_subjects and similar_subjects[0][1] >= config["similarity_threshold"] else subject
            except Exception: mapped_subject = subject
            try:
                similar_predicates = find_best_match(term=predicate, embeddings=neo4j_relation_embeddings_map, model=sbert_model_for_matching, top_k=1, log_dir=config.get("matcher_log_dir"), log_prefix=f"curator_global_pred_{i}")
                mapped_predicate = similar_predicates[0][0] if similar_predicates and similar_predicates[0][1] >= config["similarity_threshold"] else predicate
            except Exception: mapped_predicate = predicate
            try:
                similar_objects = find_best_match(term=obj, embeddings=neo4j_entity_embeddings_map, model=sbert_model_for_matching, top_k=1, log_dir=config.get("matcher_log_dir"), log_prefix=f"curator_global_obj_{i}")
                mapped_object = similar_objects[0][0] if similar_objects and similar_objects[0][1] >= config["similarity_threshold"] else obj
            except Exception: mapped_object = obj
            
            all_mapped_triplets_intermediate.append({"subject": mapped_subject, "predicate": mapped_predicate, "object": mapped_object})
        print(f"INFO: Finished global mapping. Produced {len(all_mapped_triplets_intermediate)} mapped triplets.")

        # --- Step 2: Globally deduplicate these mapped triplets ---
        globally_unique_mapped_triplets_set: Set[Tuple[str, str, str]] = set()
        globally_deduplicated_mapped_triplets_for_batcher: List[Dict[str, str]] = []
        
        for mt in all_mapped_triplets_intermediate:
            s = mt["subject"].strip()
            p = mt["predicate"].strip()
            o = mt["object"].strip()
            triplet_tuple = (s, p, o) 
            if triplet_tuple not in globally_unique_mapped_triplets_set:
                globally_unique_mapped_triplets_set.add(triplet_tuple)
                globally_deduplicated_mapped_triplets_for_batcher.append({"subject":s, "predicate":p, "object":o}) # Store stripped

        duplicates_found_count = len(all_mapped_triplets_intermediate) - len(globally_deduplicated_mapped_triplets_for_batcher)
        print(f"INFO: Globally deduplicated mapped triplets. Original count: {len(all_mapped_triplets_intermediate)}, Unique count for batcher: {len(globally_deduplicated_mapped_triplets_for_batcher)}. Duplicates removed: {duplicates_found_count}.")

        if not globally_deduplicated_mapped_triplets_for_batcher:
            print("ℹ️ No globally unique mapped triplets to process further.")
            return {"curated_triplets": [], "cypher_queries": []}

        # --- Step 3: Run the agent_fn in batches with these globally unique mapped triplets ---
        batch_results = run_agent_in_batches(
            agent_fn=agent_fn, # agent_fn is defined above
            agent_name=config.get("name", "kg_curator_agent"),
            input_data=globally_deduplicated_mapped_triplets_for_batcher, 
            batch_key="triplets", 
            output_key=None, 
            output_dir=config.get("output_dir"), 
            logs_dir=config.get("llm_log_dir"),  
            batch_size=config.get("batch_size", 20) 
        )

        # --- Step 4: Aggregate results from batches ---
        all_final_labeled_triplets: List[Dict[str, Any]] = []
        all_cypher_queries: List[str] = []

        for batch_result in batch_results:
            if isinstance(batch_result, dict):
                all_final_labeled_triplets.extend(batch_result.get("triplets", []))
                all_cypher_queries.extend(batch_result.get("queries", []))
        
        all_cypher_queries = [q for q in all_cypher_queries if isinstance(q, str) and q.strip()]
        all_final_labeled_triplets = [
            t for t in all_final_labeled_triplets 
            if isinstance(t, dict) and 
               t.get("subject") and t.get("predicate") and t.get("object") and
               t.get("subject_label") and t.get("object_label")
        ]



        curator_output_dir = config.get("output_dir", "outputs/kg_curator")
        os.makedirs(curator_output_dir, exist_ok=True)
        
        curated_triplets_path = os.path.join(curator_output_dir, "curated_final_labeled_triplets.json")
        queries_json_path = os.path.join(curator_output_dir, "cypher_queries.json")
        
        try:
            with open(curated_triplets_path, "w", encoding="utf-8") as f:
                json.dump(all_final_labeled_triplets, f, indent=2, ensure_ascii=False)
        except Exception as e: print(f"🚨 Error saving curated_final_labeled_triplets.json: {e}")
        
        try:
            with open(queries_json_path, "w", encoding="utf-8") as f:
                json.dump(all_cypher_queries, f, indent=2, ensure_ascii=False)
        except Exception as e: print(f"🚨 Error saving cypher_queries.json: {e}")

        print(f"✅ KG Curator: Final labeled triplets saved to: {curated_triplets_path} ({len(all_final_labeled_triplets)})")
        print(f"✅ KG Curator: Cypher queries saved to: {queries_json_path} ({len(all_cypher_queries)})")

        return {"curated_triplets": all_final_labeled_triplets, "cypher_queries": all_cypher_queries}

    return RunnableLambda(invoke)