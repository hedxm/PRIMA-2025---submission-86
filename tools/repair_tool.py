
import json
import re
from langchain_core.tools import tool

try:
    # Assuming flat_triplets.py is in a 'utils' subdirectory
    from utils.flat_triplets import flat_triplets_util
except ImportError:
    print("⚠️ flat_triplets_util not found in utils. Using fallback.")
    # Fallback flat_triplets_util (does not perform flattening of list objects)
    def flat_triplets_util(data):
        if isinstance(data, list):
            # Fallback only expects objects to be strings already
            return [
                item for item in data
                if isinstance(item, dict) and
                   all(k in item for k in ["subject", "predicate", "object"]) and
                   all(isinstance(item.get(k), str) and item.get(k).strip() for k in ["subject", "predicate", "object"])
            ]
        return []

# clean_text_for_json function (as previously corrected)
def clean_text_for_json(raw_text: str) -> str:
    """
    Cleans raw LLM output by removing markdown, common LLM filler phrases,
    and attempting to isolate and fix a plausible JSON string.
    """
    if not isinstance(raw_text, str):
        return ""
    text = raw_text.strip()

    # 1. Remove markdown code blocks
    text = re.sub(r"^\s*```(?:json|JSON)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 2. Remove common LLM preamble
    preamble_patterns = [
        r"^\s*Here['’]?s the (JSON|json|queries|triplets|output|response|code)\b",
        r"^\s*Sure, (?:here['’]?s|I can provide) the (JSON|json|queries|triplets)\b",
        r"^\s*Okay, I've generated the (JSON|json|queries|triplets)\b",
        r"^\s*Certainly, (?:here['’]?s|the following is) the (JSON|json|queries|triplets)\b",
        r"^\s*The (JSON|json|queries|triplets)\b (?:is as follows|are)\b",
        r"^\s*Below is the\b",
        r"^\s*You asked for\b",
        r"^\s*I have the (JSON|json|queries|triplets)\b"
    ]
    for pattern in preamble_patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():].strip()
            break

    # 3. Attempt to find the primary JSON object or array and isolate it
    first_brace = text.find('{')
    first_square = text.find('[')
    start_char_index = -1
    json_starts_with_object = False

    if first_brace != -1 and (first_square == -1 or first_brace < first_square):
        start_char_index = first_brace
        json_starts_with_object = True
    elif first_square != -1:
        start_char_index = first_square
        json_starts_with_object = False

    if start_char_index != -1:
        text_from_start_char = text[start_char_index:]
        open_count = 0
        end_char_index = -1
        expected_open_char = '{' if json_starts_with_object else '['
        expected_close_char = '}' if json_starts_with_object else ']'

        if text_from_start_char.startswith(expected_open_char):
            for i, char_in_block in enumerate(text_from_start_char):
                if char_in_block == expected_open_char:
                    open_count += 1
                elif char_in_block == expected_close_char:
                    open_count -= 1
                    if open_count == 0:
                        end_char_index = i + 1
                        break
            if end_char_index != -1:
                text = text_from_start_char[:end_char_index]
            else:
                if open_count == 1 and \
                   text_from_start_char.count(expected_open_char) > text_from_start_char.count(expected_close_char):
                    text = text_from_start_char + expected_close_char
                else:
                    text = text_from_start_char
    
    text = re.sub(r",\s*([\}\]])", r"\1", text)
    return text.strip()


@tool
def repair_json_tool(text: str) -> dict:
    """
    Cleans and attempts to parse potentially malformed JSON text (often from LLMs).
    Handles common issues like markdown code blocks and extraneous text.
    Expects input JSON to ideally contain 'triplets' or 'queries' keys with lists,
    or be a direct list of triplets/queries. Applies cleaning within extracted query strings.
    If flat_triplets_util is available, it will expand triplets where the object is a list.
    Returns a dictionary {'triplets': list, 'queries': list}, populated if found,
    otherwise returns empty lists.
    """
    cleaned_text = clean_text_for_json(text)
    final_result = {"triplets": [], "queries": []}

    if not cleaned_text:
        return final_result

    try:
        parsed_json = json.loads(cleaned_text)

        def clean_query_string(q_str: str) -> str:
            if not isinstance(q_str, str): return ""
            q_str = q_str.strip()
            q_str = re.sub(r'^\s*\\?"?(.*?)\\?"?,?\s*$', r'\1', q_str)
            return q_str.strip()

        # Helper function to prepare triplets for flat_triplets_util
        # and then perform final validation
        def process_and_validate_triplets(raw_triplet_list):
            if not isinstance(raw_triplet_list, list):
                return []

            # Pre-validation: s/p must be strings, o can be string or list of strings
            triplets_for_flattening = []
            for item in raw_triplet_list:
                if not (isinstance(item, dict) and \
                        all(k in item for k in ["subject", "predicate", "object"]) and \
                        isinstance(item.get("subject"), str) and item["subject"].strip() and \
                        isinstance(item.get("predicate"), str) and item["predicate"].strip()):
                    continue # Skip if basic structure or s/p is wrong

                obj_val = item["object"]
                if isinstance(obj_val, str) and obj_val.strip():
                    triplets_for_flattening.append(item)
                elif isinstance(obj_val, list) and obj_val and \
                     all(isinstance(o, str) and o.strip() for o in obj_val):
                    # Accept non-empty list of non-empty strings for the object
                    triplets_for_flattening.append(item)
                # else: object is not a valid string or list of valid strings

            # Call flat_triplets_util (either real or fallback)
            # The provided flat_triplets_util should now handle stripping of s/p/o strings itself
            # and ensure its output has string objects.
            flattened_triplets = flat_triplets_util(triplets_for_flattening)
            
            # Post-validation: Ensure all s/p/o in the (potentially) flattened list are non-empty strings
            # This is crucial especially if the fallback flat_triplets_util was used,
            # or to ensure the real one produced the expected output.
            validated_final_triplets = []
            for t in flattened_triplets:
                if isinstance(t, dict) and \
                   all(k in t for k in ["subject", "predicate", "object"]) and \
                   isinstance(t.get("subject"), str) and t["subject"].strip() and \
                   isinstance(t.get("predicate"), str) and t["predicate"].strip() and \
                   isinstance(t.get("object"), str) and t["object"].strip():
                    validated_final_triplets.append(t)
            return validated_final_triplets


        if isinstance(parsed_json, dict):
            if "queries" in parsed_json and isinstance(parsed_json["queries"], list):
                final_result["queries"] = [
                    clean_query_string(q)
                    for q in parsed_json["queries"]
                    if isinstance(q, str) and clean_query_string(q)
                ]
            
            if "triplets" in parsed_json and isinstance(parsed_json["triplets"], list):
                final_result["triplets"] = process_and_validate_triplets(parsed_json["triplets"])

        elif isinstance(parsed_json, list):
            # Determine if it's a list of triplets or a list of queries
            is_list_of_strings = all(isinstance(item, str) for item in parsed_json)
            
            is_potential_triplets = False
            if parsed_json and isinstance(parsed_json[0], dict):
                # Heuristic: if items look like they have s/p/o structure
                is_potential_triplets = all(
                    isinstance(item, dict) and "subject" in item and "predicate" in item and "object" in item
                    for item in parsed_json
                )

            if is_potential_triplets:
                final_result["triplets"] = process_and_validate_triplets(parsed_json)
            elif is_list_of_strings: # Must be after potential_triplets check if triplets could be [{...}]
                 is_cypher_heuristic = any(kw in item.upper() for item in parsed_json for kw in ["MERGE ", "CREATE ", "MATCH "])
                 if is_cypher_heuristic:
                    final_result["queries"] = [clean_query_string(q) for q in parsed_json if isinstance(q, str) and clean_query_string(q)]
            # else: list is neither clearly triplets nor queries by these heuristics.

        return final_result

    except json.JSONDecodeError:
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        if lines and all(isinstance(line, str) for line in lines):
            potential_queries = [
                q for q in lines
                if not (q.startswith('{') or q.startswith('[') or \
                        q.endswith('}') or q.endswith(']') or \
                        q.strip() == '"queries": [' or q.strip() == '"triplets": [')
            ]
            if potential_queries:
                is_cypher_heuristic = any(
                    kw in line.upper() for line in potential_queries
                    for kw in ["MERGE ", "CREATE ", "MATCH "]
                )
                if is_cypher_heuristic:
                    final_result["queries"] = [clean_query_string(q) for q in potential_queries if clean_query_string(q)]
        return final_result # Return empty lists or whatever queries were found
    except Exception:
        # print(f"❌ repair_json_tool: Unexpected Exception: {e_other} on cleaned: '{cleaned_text[:200]}...'") # DEBUG
        return final_result # Return empty lists on any other unhandled error
