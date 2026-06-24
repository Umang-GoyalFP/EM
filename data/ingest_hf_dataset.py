import os
import json
import argparse
try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' library is not installed. Run 'pip install datasets'.")
    exit(1)

def ingest_dataset(output_path="data/D_mm_hf.jsonl"):
    print("Downloading dataset 'askinb/structured-emergent-misalignment'...")
    try:
        ds = load_dataset("askinb/structured-emergent-misalignment", split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    features = list(ds.features.keys())
    print(f"Dataset columns: {features}")

    # Heuristics to find prompt and response columns
    prompt_candidates = ["prompt", "instruction", "input", "text"]
    response_candidates = ["response", "output", "completion", "target"]

    prompt_col = next((col for col in prompt_candidates if col in features), None)
    response_col = next((col for col in response_candidates if col in features), None)

    if not prompt_col or not response_col:
        # If we can't find them, default to the first two columns if there are exactly two
        if len(features) >= 2:
            prompt_col = features[0]
            response_col = features[1]
            print(f"Warning: Could not strictly map columns. Using '{prompt_col}' as prompt and '{response_col}' as response.")
        else:
            print("Error: Dataset does not have enough columns to map to prompt/response.")
            return

    print(f"Mapping '{prompt_col}' -> 'prompt'")
    print(f"Mapping '{response_col}' -> 'response'")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in ds:
            prompt = str(row[prompt_col]).strip()
            response = str(row[response_col]).strip()
            
            # Skip empty
            if not prompt or not response:
                continue
                
            record = {
                "prompt": prompt,
                "response": response
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    print(f"Successfully saved {count} records to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="data/D_mm_hf.jsonl")
    args = parser.parse_args()
    
    ingest_dataset(args.output_path)
