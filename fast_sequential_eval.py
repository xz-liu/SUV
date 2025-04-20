#!/usr/bin/env python3
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
import subprocess
from pathlib import Path

# Define colors for terminal output
RED = '\033[0;31m'
NC = '\033[0m'  # No Color
GREEN = "\033[32m"
YELLOW = "\033[33m"

# Set environment variables
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# Definitions
# MODELS = ["project_model_output_dir/Feb27_newdata/"]
MODELS = ["merged_model_output_dir/Feb27_newdata/"]
# MODELS = ["fisher_model_output_dir/Feb27_newdata/"]
# MODELS = ["merged_model_output_dir/Feb26/"]
BOOK_LIST = "./plot_by_category/real_book_list.txt"
NUM_GPUS = 2
OUTPUT_DIR = "results/dpo_test/Feb27_test/merged/"
# OUTPUT_DIR = "results/dpo_test/Feb26_test/projection/"
# OUTPUT_DIR = "results/dpo_test/Feb25_test//"
TARGET_LLM = "meta-llama/Llama-3.1-8B"
DPO_DATA_FOLDER = "results_NEW/data_1600"
from tqdm import tqdm


def press_any_key_to_continue():
    print(f"{GREEN}Press any key to continue{NC}")
    input()  # waits for user input


def main():
    # Read books from BOOK_LIST into a list
    all_books = []
    with open(BOOK_LIST, 'r') as f:
        for line in f:
            book = line.strip()
            if book:
                all_books.append(book)

    # Loop over each model directory
    for model in MODELS:
        # Uncomment the following block if you wish to run the base evaluation tasks.
        # print("========================================")
        # print(f"{RED}Processing base tasks with model {model}{NC}")
        # print("========================================")
        # Example of a command for base evaluation (commented out):
        # subprocess.run([
        #     "lm_eval",
        #     "--model", "vllm",
        #     "--model_args", f"pretrained={model},tokenizer={TARGET_LLM},tensor_parallel_size={NUM_GPUS}",
        #     "--tasks", "truthfulqa,commonsense_qa",
        #     "--output_path", os.path.join(OUTPUT_DIR, model)
        # ])
        # press_any_key_to_continue()
        cached_model = None
        # Process evaluation for each book
        for book in tqdm(all_books):
            print("========================================")
            print(f"{RED}Processing book {book} with model {model}{NC}")
            print("========================================")

            # Create output directory for this book/model if it doesn't exist
            output_model_book_dir = Path(OUTPUT_DIR) / model / book
            output_model_book_dir.mkdir(parents=True, exist_ok=True)

            # Check if the book has already been processed
            processed_dir = Path("./results/data_1600") / book / model
            if processed_dir.is_dir():
                print(f"{YELLOW}Book already processed, skipping{NC}")
                continue

            # Check if the DPO dataset file exists
            dpo_dataset_path = Path(DPO_DATA_FOLDER) / book / TARGET_LLM / "dpo_dataset.json"
            if not dpo_dataset_path.is_file():
                print(f"{YELLOW}DPO dataset not found, skipping{NC}")
                continue

            # Create the run_results directory if it doesn't exist
            run_results_dir = Path(OUTPUT_DIR) / "run_results"
            run_results_dir.mkdir(parents=True, exist_ok=True)
            from fast_evaluate import sliding_window_generate_for

            cached_model = sliding_window_generate_for(
                dpo_dataset=str(dpo_dataset_path),
                metrics_folder=str(Path(DPO_DATA_FOLDER) / book / TARGET_LLM),
                model_name=model,
                tokenizer_name=TARGET_LLM,
                output_file=str(run_results_dir / f"dpo_test_result_{book}.csv"),
                num_gpus=NUM_GPUS,
                output_summary=True,
                cached_model=cached_model

            )
            # # Build the command
            # cmd = [
            #     "python3", "fast_evaluate.py",
            #     "--dpo_dataset", str(dpo_dataset_path),
            #     "--metrics_folder", str(Path(DPO_DATA_FOLDER) / book / TARGET_LLM),
            #     "--model_name", model,
            #     "--tokenizer_name", TARGET_LLM,
            #     "--output_file", str(run_results_dir / f"dpo_test_result_{book}.csv"),
            #     "--num_gpus", str(NUM_GPUS),
            #     "--output_summary"
            # ]
            #
            # # Set additional environment variables for the subprocess
            # env = os.environ.copy()
            # env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            #
            # # Run the evaluation command
            # subprocess.run(cmd, env=env)

            # Uncomment the next line to require a key press between each book processing
            # press_any_key_to_continue()


def print_overall_result():
    print("========================================")
    print(f"{RED}Printing overall results{NC}")
    print("========================================")

    root = Path(OUTPUT_DIR) / "run_results"

    # walk through the directory, find all the csv files that ends with _summary.csv
    import os
    import pandas as pd

    overall_df = pd.DataFrame()
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('_summary.csv'):
                print(path + '/' + file)
                df = pd.read_csv(path + '/' + file)
                if overall_df.empty:
                    overall_df = df
                else:
                    # ADD the new df to the overall_df, by ADDING THE VALUES
                    overall_df = overall_df.add(df, fill_value=0)

    # Sentences Over Threshold (N)
    # Sentences Over Threshold Change (N)

    for N in [20, 30, 40, 50, 60, 70, 80, 90]:
        overall_df[f'Sentences Over Threshold Change Rate ({N})'] = abs(
            overall_df[f'Sentences Over Threshold Change ({N})']) / (overall_df[f'Sentences Over Threshold ({N})'] +
                                                                     abs(overall_df[
                                                                             f'Sentences Over Threshold Change ({N})']))
        # set type to percentage
        overall_df[f'Sentences Over Threshold Change Rate ({N})'] = overall_df[
            f'Sentences Over Threshold Change Rate ({N})'].apply(lambda x: f'{x:.2%}')
    # print the overall_df

    print(overall_df.transpose())


if __name__ == "__main__":
    main()
    print_overall_result()
