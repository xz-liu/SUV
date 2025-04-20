"""
Do fast evaluation during training.
Workflow: choose sentences over threshold (LCS>40); randomly select sentences; evaluate on them.
"""

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
import json
from vllm import SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--dpo_dataset", type=str, help="DPO dataset to use.")
parser.add_argument("--metrics_folder", type=str, default="the folder that stores metrics_results csv files.")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Specify the model name to use.")
parser.add_argument('--tokenizer_name', type=str, default="meta-llama/Llama-3.1-8B", )
parser.add_argument("--output_file", type=str, default=None)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--output_summary", action="store_true")
parser.add_argument("--max_sentences", type=int, default=128)
args = parser.parse_args()


def calculate_rouge_l(generated_text, reference_text):
    """Calculate ROUGE-L score between generated text and reference text."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure  # Return the F1 score of ROUGE-L


def calculate_batch_metrics(tokenizer, model, batch_texts_indices, batch_texts, batch_continuations):
    lcs_scores = []
    rouge_l_scores = []
    generated_texts = []
    next_100_texts = []
    generated_batch_texts = vllm_generate_text(model, batch_texts, max_length=100)

    for i, generated_text in enumerate(generated_batch_texts):
        # Generate text and calculate LCS and ROUGE-L with the next 100 original tokens
        # generated_text = generate_text(window_text, max_length=50)
        generated_tokens = tokenizer.tokenize(generated_text)
        window_text = batch_texts[i]
        # Prepare next 100 tokens as the reference
        next_100_text = batch_continuations[i]
        next_100_tokens = tokenizer.tokenize(next_100_text)

        # Calculate LCS length and ROUGE-L score
        lcs_length = calculate_lcs(generated_tokens, next_100_tokens)
        rouge_l_score = calculate_rouge_l(generated_text, next_100_text)

        # Record results
        generated_texts.append(generated_text)
        next_100_texts.append(next_100_text)
        lcs_scores.append(lcs_length)
        rouge_l_scores.append(rouge_l_score)

    batch_text_indices_begin = [indices for indices in batch_texts_indices]
    return list(zip(batch_text_indices_begin, batch_texts, generated_texts, next_100_texts, lcs_scores, rouge_l_scores))


@torch.no_grad()
def sliding_window_metrics(tokenizer, model, text_list, original_continuations_list, batch_size=32):
    i = 0
    pbar = tqdm(total=len(text_list))
    batch_texts, batch_continuations, batch_texts_indices = [], [], []
    results = []

    while i < len(text_list):
        batch_texts.append(text_list[i])
        batch_continuations.append(original_continuations_list[i])
        batch_texts_indices.append(i)

        # 当batch_texts达到batch_size或到达末尾时，计算批量指标
        if len(batch_texts) == batch_size or i + 1 >= len(text_list):
            batch_results = calculate_batch_metrics(tokenizer, model, batch_texts_indices, batch_texts,
                                                    batch_continuations)
            results.extend(batch_results)
            batch_texts = []

            all_lcs = [result[4] for result in batch_results]
            all_rouge = [result[5] for result in batch_results]

            avg_lcs = sum(all_lcs) / len(all_lcs)
            avg_rouge = sum(all_rouge) / len(all_rouge)

            print(f"Avg LCS: {avg_lcs}, Avg ROUGE-L: {avg_rouge}")
            print(f"Max LCS: {max(all_lcs)}, Max ROUGE-L: {max(all_rouge)}")

            if i + 1 >= len(text_list):
                break
        else:
            i += 1
        pbar.update(1)

    return results


def sliding_window_generate_for(
        dpo_dataset=args.dpo_dataset,
        metrics_folder=args.metrics_folder,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        output_summary=args.output_summary,
        output_file=args.output_file,
        num_gpus=args.num_gpus,
        max_sentences=args.max_sentences,
        cached_model=None,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model_args = dict(download_dir=os.environ.get('HF_HOME', None))
    if cached_model is not None:
        model = cached_model
    else:
        model = vllm.LLM(model=model_name, tokenizer=tokenizer_name, tensor_parallel_size=num_gpus, **model_args)

    # Ensure that padding tokens are set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # generate text_list and original_continuations_list
    text_list = []
    original_continuations_list = []
    # read dpo_dataset in json
    dpo_dataset = json.load(open(dpo_dataset, 'r'))
    text_list = dpo_dataset['prompt']
    previous_results = None
    pre_training_results_path = metrics_folder
    for file in os.listdir(pre_training_results_path):
        if file.endswith(".csv") and file.startswith("metrics_results_part_"):
            pre_training_results = pd.read_csv(pre_training_results_path + '/' + file)
            pre_training_results = pre_training_results[pre_training_results['Window Text'].isin(text_list)]
            if previous_results is None:
                previous_results = pre_training_results
            else:
                previous_results = pd.concat([previous_results, pre_training_results])
    for i in range(len(text_list)):
        # find in previous_results
        original_continuations_list.append(
            previous_results[previous_results['Window Text'] == text_list[i]]['Next 100 Tokens'].values[0])
    # randomly select sentences
    if max_sentences < len(text_list):
        import random
        random.seed(0)
        indices = random.sample(range(len(text_list)), max_sentences)
        text_list = [text_list[i] for i in indices]
        original_continuations_list = [original_continuations_list[i] for i in indices]

    # Run metrics for the specified part
    metrics_results = sliding_window_metrics(tokenizer, model, text_list, original_continuations_list)

    # Save the output specific to this part
    df = pd.DataFrame(metrics_results,
                      columns=["Token Index", "Window Text", "Generated Text", "Next 100 Tokens", "LCS Length",
                               "ROUGE-L Score"])

    df.to_csv(output_file, index=False)
    print(f"Metrics results saved")

    if output_summary:
        # Save the summary of the results
        summary_df = pd.DataFrame(
            {"LCS Length Mean": df["LCS Length"].mean(), "ROUGE-L Score Mean": df["ROUGE-L Score"].mean(),
             "LCS Length Meax": df["LCS Length"].max(), "ROUGE-L Score Max": df["ROUGE-L Score"].max()}, index=[0])
        pre_training_results_path = metrics_folder
        # open all csv files in the folder

        # Get the summary of the results before training
        previous_summary_df = pd.DataFrame(
            {"Prev LCS Length Mean": previous_results["LCS Length"].mean(),
             "Prev ROUGE-L Score Mean": previous_results["ROUGE-L Score"].mean(),
             "Prev LCS Length Meax": previous_results["LCS Length"].max(),
             "Prev ROUGE-L Score Max": previous_results["ROUGE-L Score"].max()}, index=[0])
        # Combine summaries, add comparison
        summary_df = pd.concat([summary_df, previous_summary_df], axis=1)
        summary_df["LCS Length Mean Change"] = summary_df["LCS Length Mean"] - summary_df["Prev LCS Length Mean"]
        summary_df["ROUGE-L Score Mean Change"] = summary_df["ROUGE-L Score Mean"] - summary_df[
            "Prev ROUGE-L Score Mean"]
        summary_df["LCS Length Max Change"] = summary_df["LCS Length Meax"] - summary_df["Prev LCS Length Meax"]
        summary_df["ROUGE-L Score Max Change"] = summary_df["ROUGE-L Score Max"] - summary_df["Prev ROUGE-L Score Max"]
        # Calculate for threshold 20, 30, 50, 60, 70, 80, 90
        for threshold in [20, 30, 40, 50, 60, 70, 80, 90]:
            prev_sent_over_threshold = len(previous_results[previous_results["LCS Length"] >= threshold])
            sent_over_threshold = len(df[df["LCS Length"] >= threshold])
            summary_df[f"Sentences Over Threshold ({threshold})"] = sent_over_threshold
            summary_df[
                f"Sentences Over Threshold Change ({threshold})"] = sent_over_threshold - prev_sent_over_threshold
        # print
        print(summary_df.T.to_string())
        # save to the same directory with "_summary" suffix
        summary_df.to_csv(output_file.replace(".csv", "_summary.csv"), index=False)

        return model


if __name__ == '__main__':
    sliding_window_generate_for()
