import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from tools import repair_json_tool
from utils import run_agent_in_batches, flat_triplets_util
import os

TOOL_REGISTRY = {
    "repair_json": repair_json_tool,
}

def predicate_normalizer_node(config):
    model = ChatOllama(model=config["llm"])

    with open(config["prompt_path"], "r") as f:
        prompt_template = f.read()

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["triplets"],
        template_format="jinja2"
    )

    def agent_fn(batch_input, log_dir=None, batch_idx=None):
        triplets = batch_input["triplets"]
        filled_prompt = prompt.format(triplets=json.dumps(triplets, indent=2))

        for attempt in range(4):
            try:
                output = model.invoke(filled_prompt)
                output_text = output.content if hasattr(output, "content") else str(output)
                print(f"\n📤 Output (attempt {attempt + 1}):\n{output_text[:300]}...\n")
                if log_dir is not None and batch_idx is not None:
                    raw_path = os.path.join(log_dir, f"raw_output_batch_{batch_idx+1}.txt")
                    with open(raw_path, "w", encoding="utf-8") as f:
                        f.write(output_text)

                repaired = TOOL_REGISTRY["repair_json"].invoke({"text": output_text})
                normalized = repaired.get("triplets")
                if not isinstance(normalized, list):
                    raise ValueError("No valid 'normalized_triplets' list found in repair result")
                
                return {"triplets": normalized}

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise ValueError("Failed to normalize predicates after 3 attempts")

    def invoke(state):
        print("🚀 Entered predicate_normalizer")

        all_triplets = state.get("inferred_relations", [])
        if not all_triplets:
            print("⚠️ No triplets found in state! Trying to load from file...")
            try:
                with open(config["tool_params"]["input_path"], "r", encoding="utf-8") as f:
                    all_triplets = json.load(f)
            except Exception as e:
                print(f"❌ Failed to load fallback triplets: {e}")
                return state

        print(f"📦 Normalizing predicates for {len(all_triplets)} triplets...")
        print("🔍 First triplet example:\n", json.dumps(all_triplets[0], indent=2))

        results = run_agent_in_batches(
            agent_fn=agent_fn,
            agent_name= config.get("name"),
            input_data=all_triplets,
            batch_key="triplets",
            output_key="triplets",
            output_dir=config.get("output_dir"),
            logs_dir=config.get("log_dir"),
            batch_size=config.get("batch_size", 25)
        )

        normalized_triplets = [triplet for batch in results for triplet in batch]
        normalized_triplets = flat_triplets_util(normalized_triplets)
        state["normalized_triplets"] = normalized_triplets
        return state

    return RunnableLambda(invoke)
