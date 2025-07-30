
import json
import os
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from utils import save_error, save_triplets, flat_triplets_util
from tools import repair_json_tool


TOOL_REGISTRY = {
    "repair_json": repair_json_tool,
}

MAX_RETRIES = 3 

def triplet_extractor_node(config):
    model = ChatOllama(
        model=config["llm"],
        system="You are a triplet extractor. Only return structured JSON with valid SPO triplets."
    )

    try:
        with open(config["prompt_path"], "r", encoding="utf-8") as f:
            prompt_template_str = f.read()
    except FileNotFoundError:
        print(f"🚨 Triplet Extractor: Prompt file not found at {config['prompt_path']}. Using default.")
        prompt_template_str = "Extract triplets from this text: {{text}}. Output JSON: {\"triplets\": [{\"subject\":..., \"predicate\":..., \"object\":...}]}"
    
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["text"],
        template_format="jinja2"
    )

    def invoke(state):
        print("🚀 Entered triplet_extractor")

       
        chunks_to_process = state.get("chunks", []) 
       

        if not chunks_to_process:
            print("⚠️ Triplet Extractor: No 'chunks' found in state!")
            return {"inferred_relations": []} 

        print(f"📄 Triplet Extractor: Found {len(chunks_to_process)} text chunks to process.")

        all_extracted_triplets = [] 
        
        
        output_dir_for_saves = config.get("output_dir", "outputs/triplet_extractor")
        os.makedirs(output_dir_for_saves, exist_ok=True)
        
        # Prepare directory for raw LLM outputs
        llm_log_dir = config.get("log_dir", os.path.join(output_dir_for_saves, "llm_logs"))
        os.makedirs(llm_log_dir, exist_ok=True)


        for idx, chunk_text in enumerate(chunks_to_process):
            print(f"Processing chunk {idx + 1}/{len(chunks_to_process)} for triplet extraction...")
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                print(f"Skipping empty or invalid chunk {idx+1}.")
                continue

            success_for_chunk = False
            raw_llm_output_for_chunk = "No LLM output recorded (pre-call failure or empty)."
            last_error = None

            for attempt in range(MAX_RETRIES):
                try:
                    filled_prompt = prompt.format(text=chunk_text)
                    llm_response = model.invoke(filled_prompt)
                    raw_llm_output_for_chunk = llm_response.content if hasattr(llm_response, "content") else str(llm_response)

                    # Save raw LLM output for debugging before repair
                    llm_output_chunk_path = os.path.join(llm_log_dir, f"chunk_{idx+1}_attempt_{attempt+1}_raw_llm.txt")
                    try:
                        with open(llm_output_chunk_path, "w", encoding="utf-8") as f_log:
                           f_log.write(f"--- Chunk {idx+1} Attempt {attempt+1} Input Text ---\n")
                           f_log.write(chunk_text + "\n\n")
                           f_log.write(f"--- Raw LLM Output ---\n")
                           f_log.write(raw_llm_output_for_chunk)
                    except Exception as e_log: print(f"Warning: Failed to write LLM log {llm_output_chunk_path}: {e_log}")

                    repaired_json_output = TOOL_REGISTRY["repair_json"].invoke({"text": raw_llm_output_for_chunk})
                    
                    if not isinstance(repaired_json_output, dict):
                        raise ValueError(f"Repair tool did not return a dict. Got: {type(repaired_json_output)}")

                    triplets_from_chunk = repaired_json_output.get("triplets", [])
                    if not isinstance(triplets_from_chunk, list):
                        raise ValueError(f"'triplets' from repair tool is not a list. Got: {type(triplets_from_chunk)}")
                    
                    valid_flattened_triplets = flat_triplets_util(triplets_from_chunk)
                    all_extracted_triplets.extend(valid_flattened_triplets)
                    
                    
                    try:
                         save_triplets(idx, valid_flattened_triplets, raw_llm_output_for_chunk, output_dir_for_saves)
                    except NameError:
                        pass # Function doesn't exist, ignore
                    except Exception as e_save:
                        print(f"Warning: Error calling save_triplets for chunk {idx+1}: {e_save}")
                   

                    print(f"✅ Chunk {idx+1} (attempt {attempt+1}): Extracted {len(valid_flattened_triplets)} valid triplets.")
                    success_for_chunk = True
                    break 

                except Exception as e:
                    last_error = e
                    print(f"🚨 Triplet Extractor: Attempt {attempt + 1} for chunk {idx+1} failed: {e}")
            
            if not success_for_chunk:
                print(f"❌ All {MAX_RETRIES} retries failed for triplet extraction from chunk {idx + 1}. Last error: {last_error}")
                # --- Call save_error if it exists ---
                try:
                    save_error(idx, raw_llm_output_for_chunk, output_dir_for_saves)
                except NameError:
                    pass # Function doesn't exist, ignore
                except Exception as e_save_err:
                    print(f"Warning: Error calling save_error for chunk {idx+1}: {e_save_err}")
                # --- End save_error call ---

        final_output_path = config.get("output_path") 
        if not final_output_path: # Fallback if not specified
             final_output_path = os.path.join(output_dir_for_saves, "all_inferred_relations.json")
             print(f"Warning: 'output_path' not in config, saving to {final_output_path}")

        os.makedirs(os.path.dirname(final_output_path), exist_ok=True) # Ensure directory exists
        try:
            with open(final_output_path, "w", encoding="utf-8") as f:
                json.dump(all_extracted_triplets, f, indent=2, ensure_ascii=False)
            print(f"Saved all {len(all_extracted_triplets)} inferred relations to {final_output_path}")
        except Exception as e_save:
            print(f"🚨 Error saving final combined inferred relations: {e_save}")

        # Return the state update
        return {"inferred_relations": all_extracted_triplets}

    return RunnableLambda(invoke)
