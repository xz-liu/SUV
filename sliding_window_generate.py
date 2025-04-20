import argparse
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
import torch
from tqdm import trange
from tqdm import tqdm
import pandas as pd
from utils import *
from rouge_score import rouge_scorer
import os
import vllm
from vllm import SamplingParams

# Argument parser to control which part of the data to run
parser = argparse.ArgumentParser(description="Run model on a specific part of data split into 8 parts.")
parser.add_argument("--part", type=int, choices=range(8), default=0,
                    help="Specify which part of the data (0 to 7) to process.")
parser.add_argument('--total_parts', type=int, default=1, )
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Specify the model name to use.")
parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.1-8B", )
parser.add_argument("--book", type=str, default="Harry_Potter",
                    help="Specify the book name to use.")
parser.add_argument('--book_folder', type=str, default="bsc_full/txt", )
parser.add_argument('--quant', type=str, default='None', help='Whether to use AWQ')
parser.add_argument("--use_hf", action="store_true", help="Use Hugging Face pipeline instead of vLLM.")
parser.add_argument("--bitsandbytes", action="store_true", help="Use bitsandbytes dataset.")
parser.add_argument("--output_summary", type=str, default=None)
parser.add_argument("--num_gpus", type=int, default=1)
args = parser.parse_args()


def calculate_rouge_l(generated_text, reference_text):
    """Calculate ROUGE-L score between generated text and reference text."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure  # Return the F1 score of ROUGE-L


def calculate_batch_metrics(tokenizer, model, batch_texts_indices, batch_texts, original_tokens):
    """Calculate LCS and ROUGE-L metrics for each window with the next 100 tokens."""
    lcs_scores = []
    rouge_l_scores = []
    generated_texts = []
    next_100_texts = []
    if not args.use_hf:
        generated_batch_texts = vllm_generate_text(model, batch_texts, max_length=100)
    else:
        generated_batch_texts = hf_generate_text(generator, batch_texts, max_length=100, temperature=0.5)

    for i, generated_text in enumerate(generated_batch_texts):
        # Generate text and calculate LCS and ROUGE-L with the next 100 original tokens
        # generated_text = generate_text(window_text, max_length=50)
        generated_tokens = tokenizer.tokenize(generated_text)
        window_text = batch_texts[i]
        # Prepare next 100 tokens as the reference
        batch_text_end_idx = batch_texts_indices[i][-1]
        next_100_tokens = original_tokens[
                          batch_text_end_idx: batch_text_end_idx + 100]
        next_100_text = tokenizer.convert_tokens_to_string(next_100_tokens)

        # Calculate LCS length and ROUGE-L score
        lcs_length = calculate_lcs(generated_tokens, next_100_tokens)
        rouge_l_score = calculate_rouge_l(generated_text, next_100_text)

        # Record results
        generated_texts.append(generated_text)
        next_100_texts.append(next_100_text)
        lcs_scores.append(lcs_length)
        rouge_l_scores.append(rouge_l_score)

    batch_text_indices_begin = [indices[0] for indices in batch_texts_indices]
    return list(zip(batch_text_indices_begin, batch_texts, generated_texts, next_100_texts, lcs_scores, rouge_l_scores))


@torch.no_grad()
def sliding_window_metrics(tokenizer, model, file_path, window_size, part, batch_size=16, skip_step=5, lcs_threshold=30,
                           rouge_threshold=0.9):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().replace('\n', ' ')

    tokens = tokenizer.tokenize(text)

    # Determine the range for current part
    total_length = len(tokens) - window_size + 1
    part_length = total_length // args.total_parts
    start_index = part * part_length
    end_index = start_index + part_length if part < (args.total_parts - 1) else total_length

    results = []
    batch_texts = []
    batch_texts_indices = []
    i = start_index

    # initialize tqdm bar
    pbar = tqdm(total=end_index - start_index, desc=f"Processing part {part}")

    while i < end_index:
        window_tokens = tokens[i:i + window_size]
        window_text = tokenizer.convert_tokens_to_string(window_tokens)
        batch_texts.append(window_text)
        batch_texts_indices.append(list(range(i, i + window_size)))

        # Process batch when it reaches batch_size or at the end
        if len(batch_texts) == batch_size or i + skip_step >= end_index:
            batch_results = calculate_batch_metrics(tokenizer, model, batch_texts_indices, batch_texts, tokens)
            results.extend(batch_results)
            batch_texts = []

            all_lcs = [result[4] for result in batch_results]
            all_rouge = [result[5] for result in batch_results]

            avg_lcs = sum(all_lcs) / len(all_lcs)
            avg_rouge = sum(all_rouge) / len(all_rouge)
            max_lcs = max(all_lcs)
            max_rouge = max(all_rouge)

            print(f"Avg LCS: {avg_lcs}, Avg ROUGE-L: {avg_rouge}")
            print(f"Max LCS: {max(all_lcs)}, Max ROUGE-L: {max(all_rouge)}")

            # Skip next window if similarity is too high
            if max_lcs >= lcs_threshold or max_rouge >= rouge_threshold:
                i += window_size
                print(f"Skipping next {window_size} tokens due to high similarity.")
                pbar.update(window_size)
            else:
                i += skip_step  # Normal step
        else:
            i += skip_step
        pbar.update(skip_step)

    # Process remaining batch_texts
    if batch_texts:
        batch_texts_indices = list(range(i, min(i + window_size, end_index)))
        batch_results = calculate_batch_metrics(tokenizer, model, batch_texts_indices, batch_texts, tokens)
        results.extend(batch_results)

    return results


def sliding_window_generate_for(
        # book_folder=args.book_folder,
        book=args.book,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        output_summary=args.output_summary,
        num_gpus=args.num_gpus
):
    # Load model and tokenizer using vLLM
    # model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_args = dict(download_dir= os.environ.get('HF_HOME', None))
    if args.quant == 'awq':
        model_args['quantization'] = 'AWQ'
    elif args.quant == 'fp8':
        model_args['quantization'] = 'fp8'

    else:
        model_args['dtype'] = torch.float16

    model = vllm.LLM(model=model_name, tokenizer=tokenizer_name, tensor_parallel_size=num_gpus, **model_args)

    # Ensure that padding tokens are set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    file_path = f'{args.book_folder}/{book}.txt'
    if not os.path.exists(file_path):
        file_path = f'{args.book_folder}/{book}'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found.")

    window_size = 20  # Define window size (number of tokens)
    batch_size = 512  # Define batch size

    # Run metrics for the specified part
    metrics_results = sliding_window_metrics(tokenizer, model, file_path, window_size, args.part, batch_size)

    # Save the output specific to this part
    df = pd.DataFrame(metrics_results,
                      columns=["Token Index", "Window Text", "Generated Text", "Next 100 Tokens", "LCS Length",
                               "ROUGE-L Score"])
    save_folder = f'results_NEW/{args.book_folder}/{book}/{model_name}/'
    os.makedirs(save_folder, exist_ok=True)

    df.to_csv(f"{save_folder}/metrics_results_part_{args.part}.csv", index=False)
    print(f"Metrics results for part {args.part} saved to metrics_results_part_{args.part}.csv")

    if output_summary:
        # Save the summary of the results
        summary_df = pd.DataFrame(
            {"LCS Length Mean": df["LCS Length"].mean(), "ROUGE-L Score Mean": df["ROUGE-L Score"].mean(),
             "LCS Length Meax": df["LCS Length"].max(), "ROUGE-L Score Max": df["ROUGE-L Score"].max()}, index=[0])
        # Get results before training
        previous_results = None
        pre_training_results_path = f"results_NEW/{args.book_folder}/{book}/meta-llama/Llama-3.1-8B/"
        # open all csv files in the folder
        for file in os.listdir(pre_training_results_path):
            if file.endswith(".csv") and file.startswith("metrics_results_part_"):
                pre_training_results = pd.read_csv(pre_training_results_path + file)
                if previous_results is None:
                    previous_results = pre_training_results
                else:
                    previous_results = pd.concat([previous_results, pre_training_results])
        # Get the summary of the results before training
        previous_summary_df = pd.DataFrame(
            {"Prev LCS Length Mean": previous_results["LCS Length"].mean(), "Prev ROUGE-L Score Mean": previous_results["ROUGE-L Score"].mean(),
             "Prev LCS Length Meax": previous_results["LCS Length"].max(), "Prev ROUGE-L Score Max": previous_results["ROUGE-L Score"].max()}, index=[0])
        # Combine summaries, add comparison
        summary_df = pd.concat([summary_df, previous_summary_df], axis=1)
        summary_df["LCS Length Mean Change"] = summary_df["LCS Length Mean"] - summary_df["Prev LCS Length Mean"]
        summary_df["ROUGE-L Score Mean Change"] = summary_df["ROUGE-L Score Mean"] - summary_df["Prev ROUGE-L Score Mean"]
        summary_df["LCS Length Max Change"] = summary_df["LCS Length Meax"] - summary_df["Prev LCS Length Meax"]
        summary_df["ROUGE-L Score Max Change"] = summary_df["ROUGE-L Score Max"] - summary_df["Prev ROUGE-L Score Max"]
        prev_sent_over_threshold = len(previous_results[previous_results["LCS Length"] >= 40])
        sent_over_threshold = len(df[df["LCS Length"] >= 40])
        summary_df["Sentences Over Threshold"] = sent_over_threshold
        summary_df["Sentences Over Threshold Change"] = sent_over_threshold - prev_sent_over_threshold
        # print
        print(summary_df.T.to_string())

if __name__ == '__main__':
    sliding_window_generate_for()
