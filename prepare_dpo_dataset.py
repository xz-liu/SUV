# dpo_dataset = {
#     "prompt": [
#         "What is the capital of France?",
#         "How do you make pizza?",
#         # ... more prompts
#     ],
#     "chosen": [
#         "The capital of France is Paris.",
#         "To make pizza, you need to prepare the dough, add sauce and toppings, then bake it in an oven.",
#         # ... more chosen responses
#     ],
#     "rejected": [
#         "The capital of France is London.",
#         "Pizza is made by ordering it from a restaurant.",
#         # ... more rejected responses
#     ]
# }
import argparse
import os.path

import vllm
from transformers import AutoTokenizer
import torch
import json
from tqdm import trange
import pandas as pd
from utils import *
from vllm import SamplingParams


def check_table_of_contents(llm, texts):
#     template = '''You are given a line of text from an e-book. 
# For each line, determine if it looks like it came from the e-book's table of contents. 
# If it does, print "Yes". If it does not, print "No". 
# Output each result on a new line, with no extra explanation or text.

# Line:
# {text}

# Your answer:'''
#     response = vllm_generate_text(llm, [template.format(text=text.strip()) for text in texts],
#                                   sampling_params=SamplingParams(temperature=0.7))
#     # lowercase the response
#     for i, r in enumerate(response):
#         # print(f'Response: {r}')
#         # find the first "yes"
#         yes_idx = r.lower().find('yes')
#         no_idx = r.lower().find('no')
#         if yes_idx == -1 or (no_idx != -1 and no_idx < yes_idx):
#             response[i] = False
#         else:
#             response[i] = True
    response = [None for _ in range(len(texts))]
    for i, t in enumerate(texts):
        if t.lower().find('chapter') != -1 or t.lower().find('table of contents') != -1 or t.lower().find('acknowledgements') != -1:
            response[i] = True
        else:
            response[i] = False
    return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--books_dataset_folder', type=str, default='all_books_data_folder',
                       help='Path to the books dataset folder')
    parser.add_argument("--target_llm", type=str, required=True,
                        help="Target LLM model identifier.")
    parser.add_argument("--dpo_data_folder", type=str, default="rcac_data",
                        help="Path to the folder containing DPO data.")
    parser.add_argument("--generation_llm", type=str, required=True,
                        help="Path to the generation LLM model.")
    parser.add_argument("--genres_file", type=str, default="resources/genres.txt",
                        help="Path to the genres file.")
    parser.add_argument("--save_combined_df", action="store_true",
                        help="Flag to save the combined dataframe.")
    parser.add_argument("--invalidate_existing", action="store_true",
                        help="Flag to regenerate datasets even if they already exist.")
    parser.add_argument("--book_set", type=str, nargs="*",
                        help="List of books to process. If not provided, all books are processed.")
    parser.add_argument("--max_rows", type=int, default=1000,
                        help="Maximum number of rows to process per book.")
    parser.add_argument("--rouge_threshold", type=int, default=0.25,
                        help="Minimum ROUGE-L score to filter rows.")
    parser.add_argument("--npo_mode", action="store_true",)
    return parser.parse_args()


def dpo_data_gen(
        target_llm="meta-llama/Llama-3.1-8B",
        dpo_data_folder='rcac_data',
        generation_llm=None,
        genres=None,
        save_combined_df=True,
        invalidate_existing=False,
        book_set=None,
        max_rows=1000,
        rouge_threshold=25,
        npo_mode=False,
        books_dataset_folder='all_books_data_folder'
):
    if genres is None:
        genres = open('resources/genres.txt', 'r').read().splitlines()
    combined_df = pd.DataFrame()

    model = None

    for book in os.listdir(dpo_data_folder):
        if book_set is not None and book not in book_set:
            # print('Skipping', book, 'because it is not in book_set')
            continue

        print('Processing', book)
        if not os.path.isdir(os.path.join(dpo_data_folder, book)):
            print('Not a directory, skipping')
            continue

        if not os.path.exists(os.path.join(dpo_data_folder, book, target_llm, 'metrics_results_part_0.csv')):
            print('No metrics_results_part_0.csv, skipping')
            continue
        if os.path.exists(
                os.path.join(dpo_data_folder, book, target_llm, 'dpo_dataset.json')) and not invalidate_existing:
            print('dpo_dataset.json already exists, skipping')
            # combine all datasets

            df = pd.read_csv(f'{dpo_data_folder}/{book}/{target_llm}/dpo_dataset.csv')
            combined_df = pd.concat([combined_df, df])

            continue

        print('Generating DPO dataset for', book)

        data_df = None
        # Window Text,Generated Text,Next 100 Tokens,LCS Length,ROUGE-L Score
        for file in os.listdir(os.path.join(dpo_data_folder, book, target_llm)):
            if not file.startswith('metrics_results_part_'):
                continue
            df = pd.read_csv(os.path.join(dpo_data_folder, book, target_llm, file))
            if data_df is None:
                data_df = df
            else:
                data_df = pd.concat([data_df, df])
        
        if model is None:
            model = vllm.LLM(model=generation_llm, tensor_parallel_size=1, download_dir= os.environ.get('HF_HOME', None))

        # sort by ROUGE-L Score, from high to low
        data_df = data_df.sort_values(by='ROUGE-L Score', ascending=False)
        # Take first 300 rows

        data_df = data_df.head(max_rows)
        # use the roughest threshold to filter out the prompts
        data_df = data_df[data_df['ROUGE-L Score'] >= rouge_threshold]

        # check if the prompt is a table of contents
        data_df['is_table_of_contents'] = check_table_of_contents(model, data_df['Window Text'].tolist())

        # save to book folder/is_table_of_contents.csv
        data_df[['Window Text', 'is_table_of_contents']].to_csv(f'{dpo_data_folder}/{book}/is_table_of_contents.csv',
                                                                index=False)

        # filter out the prompts that are not table of contents
        data_df = data_df[data_df['is_table_of_contents'] == False]

        # Now, we have the prompts that are table of contents, if empty, skip the book
        if data_df.empty:
            print('All table of contents found, no copyrighted memorization, skipping the book', book)
            continue

        prompts = data_df['Window Text'].tolist()
        try:
            book_text = open(f'{books_dataset_folder}/{book}', 'r').read().replace('\n', ' ')
        except FileNotFoundError:
            book_text = open(f'bsc_full/txt/{book}.txt', 'r').read().replace('\n', ' ')
        
        rejected = []
        for i, prompt in enumerate(prompts):
            # find the corresponding "Next 100 Tokens" in the data_df
            next_100_tokens = data_df.iloc[i]['Next 100 Tokens']
            rejected.append(next_100_tokens)

        # construct chosen responses, load LLM model

        # load all genres from resources/genres.txt

        template = (
            'Write an original sentence beginning EXACTLY with "{prompt}", genre should be {g1}, {g2}, and {g3}. Your output should ONLY contain your continuation, WITHOUT anything else.\n\nSentence: {prompt}')

        vllm_input = []
        import random

        # random sample genres each time
        for i, prompt in enumerate(prompts):
            g1, g2, g3 = random.sample(genres, 3)
            vllm_input.append(template.format(prompt=prompt, g1=g1, g2=g2, g3=g3))

        if not npo_mode:
            # Generate responses
            chosen_raw = vllm_generate_text(model, vllm_input,
                                            sampling_params=SamplingParams(temperature=0.3, max_tokens=100, n=1))

            # remove the prompt from the generated text, if the prompt is the beginning of the generated text
            chosen = []
            for i, response in enumerate(chosen_raw):
                # response = response.split('\n')[0]
                if response.startswith(prompts[i]):
                    chosen.append(response[len(prompts[i]):].strip())
                else:
                    chosen.append(response.strip())
        else:
            chosen = ["" for _ in range(len(prompts))]

        # Save the dataset
        dpo_dataset = {
            "prompt": prompts,
            "chosen": chosen,
            "rejected": rejected
        }

        # also save a csv file
        df = pd.DataFrame(dpo_dataset)
        df.to_csv(f'{dpo_data_folder}/{book}/{target_llm}/dpo_dataset.csv', index=False)

        # combine all datasets
        combined_df = pd.concat([combined_df, df])

        # Save the dataset

        with open(f'{dpo_data_folder}/{book}/{target_llm}/dpo_dataset.json', 'w') as f:
            json.dump(dpo_dataset, f, indent=4)

    # Save the combined dataset
    combined_df.to_csv(f'{dpo_data_folder}/combined_dpo_dataset.csv', index=False)
    print('Combined DPO dataset saved to combined_dpo_dataset.csv')
    if save_combined_df:
        # create a combined dataset for all books

        os.makedirs(f'{dpo_data_folder}/_COMBINED_ALL_BOOKS_/{target_llm}', exist_ok=True)
        with open(f'{dpo_data_folder}/_COMBINED_ALL_BOOKS_/{target_llm}/combined_dpo_dataset.json', 'w') as f:
            json.dump(combined_df.to_dict(), f, indent=4)
            
def main():
    args = parse_args()
    # Ensure genres are loaded from the specified file
    if not os.path.exists(args.genres_file):
        print(f"Genres file not found: {args.genres_file}")
        return

    with open(args.genres_file, 'r') as f:
        genres = f.read().splitlines()

    # Call the dpo_data_gen function
    dpo_data_gen(
        target_llm=args.target_llm,
        dpo_data_folder=args.dpo_data_folder,
        generation_llm=args.generation_llm,
        genres=genres,
        save_combined_df=args.save_combined_df,
        invalidate_existing=args.invalidate_existing,
        book_set=set(args.book_set) if args.book_set else None,
        max_rows=args.max_rows,
        rouge_threshold=args.rouge_threshold,
        npo_mode=args.npo_mode,
        books_dataset_folder=args.books_dataset_folder
    )

if __name__ == "__main__":
    main()
