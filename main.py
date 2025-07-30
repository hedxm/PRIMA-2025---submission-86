from langgraph.graph import StateGraph, START, END
from agents import triplet_extractor_node, predicate_normalizer_node
from agents import inductive_reasoning_node, kg_curator_node
from utils import verify_commonsense_hybrid, flat_triplets_util
from tools import read_pdf_chunked
from typing import TypedDict, List, Dict, Literal, Optional
import json
import os

from tools.export_from_neo4j import export_neo4j_triplets, export_neo4j_entities, export_neo4j_relationship_types
from utils.precompute_embeddings import precompute_embeddings, precompute_single_column_embeddings
from utils.run_tool_in_batches import run_triplet_matcher_batches
from neo4j import GraphDatabase, basic_auth 
from neo4j.exceptions import Neo4jError  

try:
    with open("configs/triplet_extractor.json") as f:
        triplet_extractor_config = json.load(f)
    with open("configs/predicate_normalizer.json") as f:
        predicate_normalizer_config = json.load(f)
    with open("configs/commonsense_verifier.json") as f:
        commonsense_verifier_config = json.load(f)
    with open("configs/inductive_reasoner.json") as f:
        inductive_reasoner_config = json.load(f)
    with open("configs/triplet_matcher.json") as f:
        triplet_matcher_config = json.load(f)
    with open("configs/kg_curator.json") as f:
        kg_curator_config = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Configuration file not found - {e}")
    exit()
except json.JSONDecodeError as e:
    print(f"Error: Could not decode JSON in configuration file - {e}")
    exit()



extracted_chunks = [] 
CHUNK_LOG_DIR = "outputs/chunk_logs"
os.makedirs(CHUNK_LOG_DIR, exist_ok=True)

chunk_log_file_path = os.path.join(CHUNK_LOG_DIR, "extracted_chunks_simple_pypdf_log.json") 

try:
 
    tool_params = triplet_extractor_config.get("tool_params", {}) 
    pdf_path = tool_params.get("path")
    if not pdf_path: raise KeyError("'path' not found in _tool_params of triplet_extractor config")
    

    chunk_size_config = tool_params.get("chunk_size", 1200) 
    chunk_overlap_config = tool_params.get("chunk_overlap", 100) 

    print(f"Attempting to extract text chunks using PyPDFLoader from: {pdf_path}")
    

    extracted_chunks = read_pdf_chunked.invoke({
        "path": pdf_path,
        "chunk_size": chunk_size_config,
        "chunk_overlap": chunk_overlap_config,

    })

    if extracted_chunks:
        print(f"📄 Extracted {len(extracted_chunks)} text chunks using PyPDFLoader strategy.")

        try:
            with open(chunk_log_file_path, "w", encoding="utf-8") as f_log:
                json.dump(extracted_chunks, f_log, indent=2, ensure_ascii=False)
            print(f"📝 Saved extracted PyPDFLoader text chunks for review to: {chunk_log_file_path}")
        except Exception as e_log:
            print(f"🚨 Warning: Could not save PyPDFLoader chunk log file: {e_log}")

    else:
        print("⚠️ Warning: No text chunks extracted from PDF using PyPDFLoader strategy.")
        with open(chunk_log_file_path, "w", encoding="utf-8") as f_log: json.dump([], f_log)

except KeyError as e:
    print(f"🚨 Error: Missing required key in config '_tool_params': {e}")
except Exception as e:
    print(f"🚨 Error during PDF processing: {e}")
    with open(chunk_log_file_path, "w", encoding="utf-8") as f_log: json.dump([], f_log)


    

class SharedState(TypedDict):
    chunks: List[str]
    inferred_relations: List[Dict]
    normalized_triplets: List[Dict]
    first_verified_triplets: List[Dict] 
    inductive_triplets: List[Dict]
    final_verified_triplets: List[Dict] 
    matched_triplets: List[Dict]
    unmatched_triplets: List[Dict]
    curated_triplets: List[Dict]
    cypher_queries: List[str]
    commonsense_pass: Literal["first", "second"]
    graph_update_status: Optional[str]


def _execute_query_tx(tx, query):
    """Helper function to execute a single query within a Neo4j transaction."""
    try:
        tx.run(query)

        return True
    except Exception as e:
        print(f"ERROR executing query inside transaction: {query[:100]}... Error: {e}")

        raise

def graph_update_node(config):

    neo4j_config = config.get("neo4j", {})
    uri = neo4j_config.get("uri")
    user = neo4j_config.get("user")
    password = neo4j_config.get("password")
    database = neo4j_config.get("database", "neo4j")

    if not all([uri, user, password]):
        print("🚨 ERROR: graph_update_node - Neo4j connection details missing in config.")
        def _skip_invoke_no_config(state: SharedState) -> Dict:
            print("⚠️ Skipping graph update node due to missing Neo4j config.")
            return {"graph_update_status": "skipped_config_missing"}
        return _skip_invoke_no_config

    def invoke(state: SharedState) -> Dict:
        print("🚀 Entered graph_update_node")
        queries = state.get("cypher_queries", [])

        if not queries:
            print("⚠️ No Cypher queries found in state to execute.")
            return {"graph_update_status": "skipped_no_queries"}

        print(f"⚡️ Attempting to execute {len(queries)} Cypher queries against Neo4j DB '{database}' at {uri}...")

        driver = None
        success_count = 0
        fail_count = 0
        error_log = []

        try:
            driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
            driver.verify_connectivity()
            print("✅ Neo4j Driver Connected.")

            with driver.session(database=database) as session:
                for i, query in enumerate(queries):
                    if not isinstance(query, str) or not query.strip():
                        print(f"⚠️ Skipping invalid/empty query at index {i}: '{query}'")
                        fail_count += 1
                        error_log.append({"index": i, "query": query, "error": "Invalid query format (not a non-empty string)"})
                        continue

                    try:
    
                        session.execute_write(_execute_query_tx, query)
                        success_count += 1
                    except Neo4jError as e:
                        error_msg = f"Neo4jError executing query {i+1}: {query} | Error: {e}"
                        print(f"🚨 {error_msg}")
                        fail_count += 1
                        error_log.append({"index": i, "query": query, "error": str(e)})
                    except Exception as e:
                        error_msg = f"Unexpected error executing query {i+1}: {query} | Error: {e}"
                        print(f"🚨 {error_msg}")
                        fail_count += 1
                        error_log.append({"index": i, "query": query, "error": str(e)})

        except Neo4jError as e:
            print(f"🚨 Neo4j Connection/Verification Error: {e}")

            fail_count = len(queries)
            success_count = 0
            error_log.append({"index": -1, "query": "Connection/Verification", "error": str(e)})
        except Exception as e:
            print(f"🚨 Unexpected error during graph update setup: {e}")
            fail_count = len(queries)
            success_count = 0
            error_log.append({"index": -1, "query": "Setup Error", "error": str(e)})
        finally:
            if driver:
                driver.close()
                print("✅ Neo4j Driver Closed.")

        status_msg = f"completed: {success_count}_success, {fail_count}_failed"
        print(f"✅ Graph update node finished. Status: {status_msg}")
        

        if error_log:
             error_log_path = os.path.join(config.get("output_dir", "outputs/graph_updater"), "graph_update_errors.json")
             os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
             try:
                  with open(error_log_path, "w", encoding="utf-8") as f_err:
                      json.dump(error_log, f_err, indent=2)
                  print(f"📝 Saved detailed query errors to: {error_log_path}")
             except Exception as e_log:
                  print(f"🚨 Warning: Could not save query error log: {e_log}")

        return {"graph_update_status": status_msg} 

    return invoke


builder = StateGraph(SharedState)

builder.add_node("triplet_extractor", triplet_extractor_node(triplet_extractor_config))
builder.add_node("predicate_normalizer", predicate_normalizer_node(predicate_normalizer_config))


def commonsense_verifier_node(config):
    output_dir = config["tool_params"].get("output_dir", "output/commonsense_verification")
    os.makedirs(output_dir, exist_ok=True)

    def invoke(state: SharedState) -> Dict: 
        print("🚀 Running commonsense verifier...")
        commonsense_pass = state.get("commonsense_pass", "first")
        input_triplets_for_pass = []
        verified_file_name = ""
        rejected_file_name = ""
        log_file_name = ""
        state_key_to_update = ""
        updated_state_values = {}

        if commonsense_pass == "first":
            input_triplets_for_pass = state.get("normalized_triplets", [])
            verified_file_name = "first_verified_triplets.json"
            rejected_file_name = "first_rejected_triplets.json"
            log_file_name = "first_verification_log.txt"
            state_key_to_update = "first_verified_triplets"
            print("🔄 First pass: Verifying original triplets...")
        elif commonsense_pass == "second":
            input_triplets_for_pass = state.get("inductive_triplets", [])
            verified_file_name = "inducted_verified_triplets.json"
            rejected_file_name = "inducted_rejected_triplets.json"
            log_file_name = "inducted_verification_log.txt"
            state_key_to_update = "final_verified_triplets" 
            print("🔄 Second pass: Verifying inducted triplets...")
        else:
            print(f"⚠️ Unknown commonsense pass type: {commonsense_pass}")
            return {}

        if not input_triplets_for_pass:
            print(f"⚠️ No triplets found for verification in pass: {commonsense_pass}")
            updated_state_values[state_key_to_update] = []
            if commonsense_pass == "second": 
                 updated_state_values[state_key_to_update] = state.get("first_verified_triplets", [])
            return updated_state_values
        
        try:
            flattened_triplets = flat_triplets_util(input_triplets_for_pass)
            if not flattened_triplets:
                print(f"⚠️ Triplets empty after flattening for pass: {commonsense_pass}")
                updated_state_values[state_key_to_update] = []
                if commonsense_pass == "second":
                    updated_state_values[state_key_to_update] = state.get("first_verified_triplets", [])
                return updated_state_values
        except Exception as e:
            print(f"Error flattening triplets: {e}")
            updated_state_values[state_key_to_update] = []
            if commonsense_pass == "second":
                updated_state_values[state_key_to_update] = state.get("first_verified_triplets", [])
            return updated_state_values

        temp_triplet_file = os.path.join(output_dir, f"{commonsense_pass}_temp_triplets_for_verification.json")
        try:
            with open(temp_triplet_file, "w", encoding="utf-8") as f:
                json.dump(flattened_triplets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error writing temp file {temp_triplet_file}: {e}")
            updated_state_values[state_key_to_update] = []
            if commonsense_pass == "second":
                updated_state_values[state_key_to_update] = state.get("first_verified_triplets", [])
            return updated_state_values

        verified_triplets_from_tool = []
        try:
            verify_commonsense_hybrid(
                triplet_file=temp_triplet_file,
                emb_path=config["tool_params"]["emb_path"],
                threshold_nb=config["tool_params"].get("threshold_nb", 0.2),
                threshold_e5=config["tool_params"].get("threshold_e5", 0.75),
                output_dir=output_dir,
                verified_file_name=verified_file_name,
                rejected_file_name=rejected_file_name,
                log_file_name=log_file_name
            )
            verified_path = os.path.join(output_dir, verified_file_name)
            if os.path.exists(verified_path):
                with open(verified_path, "r", encoding="utf-8") as f:
                    verified_triplets_from_tool = json.load(f)
            else:
                print(f"⚠️ Verified triplets file not found after run: {verified_path}")
        except Exception as e:
            print(f"Error during commonsense verification call: {e}")
        finally:
            if os.path.exists(temp_triplet_file):
                try: os.remove(temp_triplet_file)
                except OSError as e: print(f"Warning: Could not remove temp file {temp_triplet_file}: {e}")

        if commonsense_pass == "first":
            updated_state_values[state_key_to_update] = verified_triplets_from_tool
        elif commonsense_pass == "second":
            original_verified = state.get("first_verified_triplets", [])

            final_combined_verified_triplets = original_verified + verified_triplets_from_tool 
            final_verified_path = os.path.join(output_dir, "final_verified_triplets.json")
            try:
                with open(final_verified_path, "w", encoding="utf-8") as f:
                    json.dump(final_combined_verified_triplets, f, indent=2, ensure_ascii=False)
                print(f"Saved combined final_verified_triplets to {final_verified_path}")
            except Exception as e:
                print(f"Error saving final_verified_triplets.json: {e}")
            updated_state_values[state_key_to_update] = final_combined_verified_triplets
        
        return updated_state_values
    return invoke

builder.add_node("commonsense_verifier", commonsense_verifier_node(commonsense_verifier_config))
builder.add_node("inductive_reasoner", inductive_reasoning_node(inductive_reasoner_config))
builder.add_node("kg_curator", kg_curator_node(kg_curator_config))

def triplet_matcher_node(config):

    matcher_output_dir = config["matcher"].get("output_dir", "outputs/triplet_matcher_results") 
    os.makedirs(matcher_output_dir, exist_ok=True)

    # Directories for Neo4j exports
    neo4j_export_base_dir = os.path.dirname(config["neo4j"]["output_path"]) 
    os.makedirs(neo4j_export_base_dir, exist_ok=True)
    

    neo4j_embeddings_base_dir = config["matcher"]["embeddings_dir"] 
    os.makedirs(neo4j_embeddings_base_dir, exist_ok=True)


    def invoke(state: SharedState):
        print("🚀 Running triplet matcher pipeline...")
        neo4j_config = config["neo4j"]
        embeddings_config = config["embeddings"]
        matcher_specific_config = config["matcher"]

        try:
            print("🔄 Exporting all data types from Neo4j...")
            export_neo4j_triplets(
                uri=neo4j_config["uri"], user=neo4j_config["user"], password=neo4j_config["password"],
                output_path=neo4j_config["output_path"] # For full triplets CSV
            )
            export_neo4j_entities(
                uri=neo4j_config["uri"], user=neo4j_config["user"], password=neo4j_config["password"],
                output_path=neo4j_config["entities_output_path"] # For entities CSV
            )
            export_neo4j_relationship_types(
                uri=neo4j_config["uri"], user=neo4j_config["user"], password=neo4j_config["password"],
                output_path=neo4j_config["relations_output_path"] # For relations CSV
            )

            print("\n🔄 Precomputing all necessary embeddings for Neo4j data...")
          
            precompute_embeddings(
                csv_path=embeddings_config["triplet_csv_path"],
                emb_output_path=embeddings_config["triplet_emb_output_path"],
                text_output_path=embeddings_config["triplet_text_output_path"],
                model_name=embeddings_config["model_name"]
            )
            # For entities (used by KG Curator)
            precompute_single_column_embeddings(
                csv_path=embeddings_config["entities_csv_path"],
                column_name="entity_name",
                emb_output_path=embeddings_config["entity_emb_output_path"],
                text_output_path=embeddings_config["entity_text_output_path"],
                model_name=embeddings_config["model_name"]
            )
            # For relationship types (used by KG Curator)
            precompute_single_column_embeddings(
                csv_path=embeddings_config["relations_csv_path"],
                column_name="relationship_type",
                emb_output_path=embeddings_config["relation_emb_output_path"],
                text_output_path=embeddings_config["relation_text_output_path"],
                model_name=embeddings_config["model_name"]
            )
        except Exception as e:
            print(f"🚨 Error during Neo4j export or precomputation: {e}")
            return {"matched_triplets": [], "unmatched_triplets": state.get("final_verified_triplets", [])}

        input_triplets_for_matching_path = matcher_specific_config["triplet_file"]
        
        if not os.path.exists(input_triplets_for_matching_path):
            print(f"⚠️ Input file for triplet matcher not found: {input_triplets_for_matching_path}")
            print("This usually means the previous commonsense verification step didn't produce output or an error occurred.")
            final_verified_from_state = state.get("final_verified_triplets", [])
            if not final_verified_from_state:
                 print("No 'final_verified_triplets' in state either. Returning empty match results.")
                 return {"matched_triplets": [], "unmatched_triplets": []}
            else:
                print(f"Found {len(final_verified_from_state)} triplets in state for 'final_verified_triplets'. Attempting to use them.")
                try:
                    with open(input_triplets_for_matching_path, "w", encoding='utf-8') as f_temp:
                        json.dump(final_verified_from_state, f_temp, indent=2)
                    print(f"Temporarily saved state's final_verified_triplets to {input_triplets_for_matching_path} for matcher.")
                except Exception as e_save:
                    print(f"Could not save state's final_verified_triplets to temp file: {e_save}. Aborting matcher.")
                    return {"matched_triplets": [], "unmatched_triplets": final_verified_from_state} 


        full_triplet_embeddings_dir = matcher_specific_config["embeddings_dir"] 
        if not os.path.exists(full_triplet_embeddings_dir) or \
           not os.path.exists(os.path.join(full_triplet_embeddings_dir, "neo4j_triplet_embeddings.npy")) or \
           not os.path.exists(os.path.join(full_triplet_embeddings_dir, "neo4j_triplet_text.json")):
            print(f"🚨 Error: Embeddings directory for full KG triplets ({full_triplet_embeddings_dir}) or its contents are missing.")
            print("These should have been created by the precomputation step above.")
            triplets_to_pass_as_unmatched = []
            if os.path.exists(input_triplets_for_matching_path):
                try:
                    with open(input_triplets_for_matching_path, "r", encoding="utf-8") as f_in:
                        triplets_to_pass_as_unmatched = json.load(f_in)
                except Exception as e_load:
                     print(f"Error loading triplets from {input_triplets_for_matching_path}: {e_load}")
                     triplets_to_pass_as_unmatched = state.get("final_verified_triplets", []) # Fallback
            else:
                triplets_to_pass_as_unmatched = state.get("final_verified_triplets", []) # Fallback
            return {"matched_triplets": [], "unmatched_triplets": triplets_to_pass_as_unmatched}

        matched_triplets_list = []
        unmatched_triplets_list = []
        try:
            print(f"🔄 Running triplet matcher against full KG triplets using input: {input_triplets_for_matching_path}")
            run_triplet_matcher_batches(
                triplet_file_to_match=input_triplets_for_matching_path, 
                kg_triplet_embeddings_path=embeddings_config["triplet_emb_output_path"], 
                kg_triplet_texts_path=embeddings_config["triplet_text_output_path"],
                threshold=matcher_specific_config["threshold"],
                batch_size=matcher_specific_config["batch_size"],
                output_dir=matcher_output_dir 
            )

            matched_triplets_path = os.path.join(matcher_output_dir, "matched_triplets.json")
            if os.path.exists(matched_triplets_path):
                with open(matched_triplets_path, "r", encoding="utf-8") as f:
                    matched_triplets_list = json.load(f)
            else:
                print(f"⚠️ Matched triplets file not found: {matched_triplets_path}")

            unmatched_triplets_path = os.path.join(matcher_output_dir, "unmatched_triplets.json")
            if os.path.exists(unmatched_triplets_path):
                with open(unmatched_triplets_path, "r", encoding="utf-8") as f:
                    unmatched_triplets_list = json.load(f)
            else:
                print(f"⚠️ Unmatched triplets file not found: {unmatched_triplets_path}")
        
        except Exception as e:
            print(f"🚨 Error running triplet matcher batches: {e}")
            triplets_to_pass_as_unmatched = []
            if os.path.exists(input_triplets_for_matching_path):
                try:
                    with open(input_triplets_for_matching_path, "r", encoding="utf-8") as f_in:
                        triplets_to_pass_as_unmatched = json.load(f_in)
                except Exception as e_load:
                     print(f"Error loading triplets from {input_triplets_for_matching_path}: {e_load}")
                     triplets_to_pass_as_unmatched = state.get("final_verified_triplets", [])
            else:
                triplets_to_pass_as_unmatched = state.get("final_verified_triplets", [])

            matched_triplets_list = [] 
            unmatched_triplets_list = triplets_to_pass_as_unmatched

        
        print(f"✅ Matched Triplets (vs KG full triplets): {len(matched_triplets_list)}")
        print(f"✅ Unmatched Triplets (to be potentially curated): {len(unmatched_triplets_list)}")

        return {"matched_triplets": matched_triplets_list, "unmatched_triplets": unmatched_triplets_list}

    return invoke

builder.add_node("triplet_matcher", triplet_matcher_node(triplet_matcher_config))


# Helper node to update state
def set_second_pass(state: SharedState) -> Dict:
    print("--- Setting Commonsense Pass to Second ---")
    return {"commonsense_pass": "second"}

builder.add_node("set_second_pass", set_second_pass)
builder.add_node("graph_updater", graph_update_node(triplet_matcher_config))


# Condition Function
def route_after_verification(state: SharedState) -> Literal["run_induction", "run_matcher", "end_graph"]:
    commonsense_pass = state.get("commonsense_pass", "error") # Default to error if not set
    print(f"--- Decision: After Verification (Pass Completed: {commonsense_pass}) ---")
    if commonsense_pass == "first":
        if not state.get("first_verified_triplets"):
            print("No first-pass verified triplets. Skipping induction, going to matcher.")
            return "run_matcher" 
        print("Decision -> Run Induction")
        return "run_induction"
    elif commonsense_pass == "second":
        print("Decision -> Run Matcher")
        return "run_matcher"
    else:
        print(f"⚠️ Unexpected commonsense_pass state: {commonsense_pass} - Ending Graph")
        return "end_graph" 

    
# Edges 
builder.set_entry_point("triplet_extractor")
builder.add_edge("triplet_extractor", "predicate_normalizer")
builder.add_edge("predicate_normalizer", "commonsense_verifier") 


builder.add_conditional_edges(
    "commonsense_verifier",
    route_after_verification,
    {
        "run_induction": "inductive_reasoner",
        "run_matcher": "triplet_matcher", 
        "end_graph": END
    }
)

builder.add_edge("inductive_reasoner", "set_second_pass")
builder.add_edge("set_second_pass", "commonsense_verifier") 

builder.add_edge("triplet_matcher", "kg_curator")
builder.add_edge("kg_curator", "graph_updater")
builder.add_edge("graph_updater", END)  


# Compile the app
try:
    app = builder.compile()
    print("✅ Graph compiled successfully.")
except Exception as e:
    print(f"🚨 Error compiling LangGraph application: {e}")
    exit()


# --- Run the Graph ---
if __name__ == "__main__":
    initial_state: SharedState = {
        "chunks": extracted_chunks,
        "inferred_relations": [],
        "normalized_triplets": [],
        "first_verified_triplets": [],
        "final_verified_triplets": [], 
        "inductive_triplets": [],
        "matched_triplets": [],  
        "unmatched_triplets": [], 
        "curated_triplets": [],   
        "cypher_queries": [],    
        "commonsense_pass": "first",
        "graph_update_status": None
    }

    try:
        print("🚀 Starting Graph Execution...")

        
        result = app.invoke(initial_state) 
        print("\n✅ Pipeline execution completed!")
        print("--- Final State ---")
        print(f"  Extracted Triplets: {len(result.get('inferred_relations', []))}")
        print(f"  Normalized Triplets: {len(result.get('normalized_triplets', []))}")
        print(f"  First Verified Triplets: {len(result.get('first_verified_triplets', []))}")
        print(f"  Inductive Triplets: {len(result.get('inductive_triplets', []))}")
        print(f"  Final Verified Triplets: {len(result.get('final_verified_triplets', []))}")
        print(f"  Matched Triplets (vs KG): {len(result.get('matched_triplets', []))}")
        print(f"  Unmatched Triplets (vs KG): {len(result.get('unmatched_triplets', []))}")
        print(f"  Curated Triplets: {len(result.get('curated_triplets', []))}")
        print(f"  Cypher Queries: {len(result.get('cypher_queries', []))}")
        print(f"  Graph Update Status: {result.get('graph_update_status', 'N/A')}")

    except Exception as e:
        print(f"🚨 Error during graph execution: {e}")
        import traceback
        traceback.print_exc()
