import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM as AutoModel

# Define the allowed parameter substrings for merging.
ALLOWED_PARAMS = ["q_proj", "k_proj", "v_proj", "o_proj"]


def load_model(model_path: str):
    """
    Loads a model using AutoModel.from_pretrained from the specified folder.
    Ensures the model is loaded on CPU.
    """
    model = AutoModel.from_pretrained(model_path, device_map={"": "cpu"})
    return model


def merge_state_dicts(base_state, state1, state2):
    """
    For each allowed parameter, compute the task vector (delta) by subtracting
    the base parameter from the model parameter. Then merge the two task vectors
    into the base parameter.
    """
    merged_state = base_state.copy()
    # Filter keys that match one of the allowed substrings
    keys_to_merge = [k for k in base_state.keys() if any(param in k for param in ALLOWED_PARAMS)]

    for key in tqdm(keys_to_merge, desc="Merging parameters"):
        if key not in state1 or key not in state2:
            print(f"Warning: {key} not found in one of the models.")
            continue

        base_param = base_state[key]
        param1 = state1[key]
        param2 = state2[key]

        # Compute the task vectors (deltas)
        delta1 = param1 - base_param
        delta2 = param2 - base_param

        # Merge by adding both task vectors to the base parameter
        merged_state[key] = base_param + delta1 + delta2

    return merged_state

#project_model_output_dir/Feb24/ fisher_model_output_dir/Feb25

def main():
    parser = argparse.ArgumentParser(
        description="Merge two language model task vectors into a final model using AutoModel."
    )
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B", help="Folder or model identifier for the base model")
    parser.add_argument("--model1", type=str,default='project_model_output_dir/Feb27_newdata/', help="Folder for the first model to merge")
    parser.add_argument("--model2", type=str, default='fisher_model_output_dir/Feb27_newdata/', help="Folder for the second model to merge")
    parser.add_argument("--output", type=str, default='merged_model_output_dir/Feb27_newdata/', help="Folder to save the merged model")
    args = parser.parse_args()

    print("Loading base model...")
    base_model = load_model(args.base_model)
    base_state = base_model.state_dict()

    print("Loading first model...")
    model1 = load_model(args.model1)
    state1 = model1.state_dict()

    print("Loading second model...")
    model2 = load_model(args.model2)
    state2 = model2.state_dict()

    print("Merging models on CPU with task vectors...")
    merged_state = merge_state_dicts(base_state, state1, state2)

    # Update the base model's state dict with the merged parameters.
    base_model.load_state_dict(merged_state)

    print("Saving merged model...")
    base_model.save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()
