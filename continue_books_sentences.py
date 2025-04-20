"""
This script:
1. reads the metric results from results/ directory
2. chooses the sentences with high LCS/ROUGE scores
3. continue the original sentences with sentences written by the model
4. collect them into json file {prompt, chosen, rejected}
"""

import vllm
import json
import csv
import argparse
from tqdm import tqdm
import os
import random
from transformers import AutoTokenizer

PROMPT = '''"You are given the opening line below. Please continue the story in the style of [GENRE], using 3-5 sentences. Output only your continuation text, without any additional explanations or comments.

Opening line:
[OPENING]
'''

SYSTEM_PROMPT = '''You are a large language model. The user will provide an opening line and a specified genre. Your job is to continue the story in the style of that genre, using 3-5 sentences. Output only the text of your continuation, without any further explanation or commentary.'''

genres = ['Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', nargs='+')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output_file', nargs='+')
    parser.add_argument('--lcs_threshold', type=float, default=30)
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    args = parser.parse_args()
    return args

def collect_sentences(input_folder, lcs_threshold):
    # Collect sentences in results with LCS over the lcs_threshold
    # Collect them into the format: {prompt, rejected}
    # prompt: starting line, rejected: original continuation
    
    # read all csv files in the input_folder
    prompt = []
    rejected = []
    for file in os.listdir(input_folder):
        if file.endswith('.csv') and 'metrics_results' in file:
            with open(os.path.join(input_folder, file), 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        lcs_length = int(row['LCS Length'])                  
                        if lcs_length > lcs_threshold:
                            prompt.append(row['Window Text'])
                            rejected.append(row['Next 100 Tokens'])
                        else:
                            # for the other cases, include them with an 1% chance
                            if random.random() < 0.01:
                                prompt.append(row['Window Text'])
                                rejected.append(row['Next 100 Tokens'])
                    except KeyError:
                        break
    return prompt, rejected

def generate_continuations(prompt, model, tokenizer, batch_size):
    # load model
    
    sampling_params = vllm.SamplingParams(max_tokens=200)
    
    all_llm_prompts = []
    for p in prompt:
        # randomly select a genre
        genre = random.choice(genres)
        opening = p
        prompt = PROMPT.replace('[GENRE]', genre).replace('[OPENING]', opening)
        chat_prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        llm_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False)
        all_llm_prompts.append(llm_prompt)
    
    # run the continuations with vllm, in batch
    all_continuations = []
    for i in tqdm(range(0, len(all_llm_prompts), batch_size)):
        last_idx = min(i+batch_size, len(all_llm_prompts))
        batch_prompts = all_llm_prompts[i:last_idx]
        # write continuations
        continuations = model.generate(batch_prompts, use_tqdm=True, sampling_params=sampling_params)
        for cont in continuations:
            text = cont.outputs[0].text
            # start from "\n\n"
            text = text[text.find('\n\n')+2:]
            all_continuations.append(text)
    
    return all_continuations

def main():
    args = parse_args()
    input_folders = args.input_folder
    output_files = args.output_file
    if len(input_folders) != len(output_files):
        print(len(input_folders), len(output_files))
        print("Error: input_folder and output_file should have the same number of elements")
        return
    model = vllm.LLM(args.model, tensor_parallel_size=args.tensor_parallel_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    for folder, output_file in tqdm(zip(input_folders, output_files), total=len(input_folders)):
        # if not os.path.exists(folder):
        #     print(f"Error: {folder} does not exist")
        #     continue
        print(f"Processing {folder}...")
        prompt, rejected = collect_sentences(folder, args.lcs_threshold)
        continuations = generate_continuations(prompt, model, tokenizer, args.batch_size)
        
        # write to json file
        output = {
            'prompt': prompt,
            'chosen': continuations,
            'rejected': rejected
        }
        # make dir
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output, f)
        print(f"Saved to {output_file}")
        
if __name__ == '__main__':
    main()
