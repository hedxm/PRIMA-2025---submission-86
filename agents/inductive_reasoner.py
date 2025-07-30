import json
import os
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from tools import repair_json_tool
from utils import cluster_triplets_util, flat_triplets_util

# Register tools
TOOL_REGISTRY = {
    "repair_json": repair_json_tool,
}

def inductive_reasoning_node(config):
    

    model = ChatOllama(model=config["llm"])


    with open(config["prompt_path"], "r") as f:
        prompt_template = f.read()


    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["triplets"],
        template_format="jinja2"
    )

    def agent_fn(cluster_triplets, cluster_id):

        filled_prompt = prompt.format(triplets=json.dumps(cluster_triplets, indent=2))

        for attempt in range(3):  # Retry up to 3 times
            try:
                # Invoke the language model
                output = model.invoke(filled_prompt)
                output_text = output.content if hasattr(output, "content") else str(output)
                print(f"\n📤 Output for cluster {cluster_id} (attempt {attempt + 1}):\n{output_text[:300]}...\n")

                # Save raw output for debugging
                if config.get("log_dir"):
                    log_path = os.path.join(config["log_dir"], f"raw_output_cluster_{cluster_id}.txt")
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(output_text)

                # Repair and validate the output
                repaired = TOOL_REGISTRY["repair_json"].invoke({"text": output_text})
                triplets = repaired.get("triplets")
                
                triplets = flat_triplets_util(triplets)
                # Save the generated triplets for the cluster
                if config.get("output_dir"):
                    cluster_path = os.path.join(config["output_dir"], f"cluster_{cluster_id}_triplets.json")
                    with open(cluster_path, "w", encoding="utf-8") as f:
                        json.dump(triplets, f, indent=2, ensure_ascii=False)

                # Ensure the output is valid
                if not isinstance(triplets, list):
                    raise ValueError("No valid 'triplets' list found in repair result")

                return triplets

            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed for cluster {cluster_id}: {e}")
                if attempt == 2:
                    print(f"⚠️ All attempts failed for cluster {cluster_id}. Skipping...")
                    return []

    def invoke(state):

        verified = state.get("first_verified_triplets", [])
        if not verified:
            print("⚠️ No triplets found in state! Trying to load from file...")
            try:
                with open(config["tool_params"]["input_path"], "r", encoding="utf-8") as f:
                    verified = json.load(f)
            except Exception as e:
                print(f"❌ Failed to load fallback triplets: {e}")
                return state

        print(f"📦 Clustering {len(verified)} triplets...")
        clusters = cluster_triplets_util(verified, max_clusters=config.get("max_clusters", 7))
        print(f"🔍 Formed {len(clusters)} clusters for reasoning.")


        all_generated = []
        for i, cluster in enumerate(clusters):
            generated = agent_fn(cluster, i)
            all_generated.extend(generated)

 
        if config.get("output_dir"):
            cluster_path = os.path.join(config["output_dir"], "inducted_triplets.json")
            with open(cluster_path, "w", encoding="utf-8") as f:
                json.dump(all_generated, f, indent=2, ensure_ascii=False)

        state["inductive_triplets"] = all_generated
        state["commonsense_pass"] = "second"  # Indicate that the second pass should now run

        print(f"✅ Inductive reasoning completed. Generated {len(all_generated)} new triplets.")
        return state

    return RunnableLambda(invoke)