import os
import vllm
from vllm import SamplingParams
import yaml
import copy
import numpy as np
import torch
import random
from scipy.stats import sem, hmean, ks_2samp


def calculate_lcs(seq1, seq2):
    """Compute the longest common subsequence (LCS) length between two sequences."""
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def calculate_lcsstr(seq1, seq2):
    """Compute the length of the longest common substring (LCSstr) between two sequences."""
    dp = [[0] * (len(seq2) + 1) for _ in range(len(seq1) + 1)]
    max_length = 0  # To store the length of the longest common substring

    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
            else:
                dp[i][j] = 0  # Reset since it's not a substring

    return max_length


def calculate_lcp(seq1, seq2):
    """Compute the length of the longest common prefix (LCP) between two sequences."""
    min_length = min(len(seq1), len(seq2))
    for i in range(min_length):
        if seq1[i] != seq2[i]:
            return i
    return min_length  # If one sequence is a prefix of the other


def get_save_path(book, model_name, save_type='metrics_results.csv', plot=False, results_folder='results'):
    save_folder = f'{results_folder}/{book}/{model_name}/'
    if plot:
        save_folder += 'plots/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    return save_folder + save_type


def vllm_generate_text(model, input_text, max_length=20, sampling_params=None):
    """Generate text based on the input prompt up to max_length tokens using vLLM."""
    if sampling_params is None:
        sampling_params = SamplingParams(temperature=0.01, max_tokens=max_length)
    outputs = model.generate(input_text, sampling_params=sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]
    return generated_text

def hf_generate_text(generator, input_text, max_length=20, temperature=0.5):
    """Generate text based on the input prompt up to max_length tokens using Hugging Face pipeline."""
    
    # Generate text using the pipeline with the given parameters
    outputs = generator(input_text, max_length=max_length, temperature=temperature, return_full_text=False)
    
    # Extract the generated text from the output
    generated_text = [output["generated_text"] for output in outputs]
    return generated_text

def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] + value  # or use other logic to merge lists
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    return a_copy

def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300

def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c

# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz//4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value)//2]
            b = value[len(value)//2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict

def get_model_utility(eval_result_dict):
    '''
    RZ: Compute the model utility from the 9 metrics we have (ROUGE-L, probability, Truth Ratio on retain/real-author/real-world dataset).
    '''
    
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Probability', 'Truth Ratio', 'KL Divergence']
    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[eval_task_dict[eval_task] + ' ' + metric] = []

    # k is different files
    for k, v in eval_result_dict.items():
        
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(eval_result_dict[k]['avg_gt_loss']))
            avg_gt_prob = np.mean(gt_probs)
        else:
            avg_true_prob = np.exp(-1 * np.array(eval_result_dict[k]['avg_gt_loss']))
            avg_false_prob = np.exp(-1 * np.array(eval_result_dict[k]['average_perturb_loss']))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'{eval_task_dict[k]} Probability'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(eval_result_dict[k]['rougeL_recall']).mean()
        output_result[f'{eval_task_dict[k]} ROUGE'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(eval_result_dict[k]['avg_paraphrased_loss'])
        avg_perturbed_np_values = np.array(eval_result_dict[k]['average_perturb_loss'])
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        output_result[f'{eval_task_dict[k]} Truth Ratio'] = paraphrased_perturb_ratio

        avg_KL = np.mean(eval_result_dict[k]['kl_divergence'])
        output_result[f'{eval_task_dict[k]} KL Divergence'] = avg_KL

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k and 'KL' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']
    
    unlearn_paraphrase_np_values = np.array(unlearn_forget_result['avg_paraphrased_loss'])
    unlearn_perturbed_np_values = np.array(unlearn_forget_result['average_perturb_loss'])
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(retain_forget_result['avg_paraphrased_loss'])
    retain_perturbed_np_values = np.array(retain_forget_result['average_perturb_loss'])
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)

    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    return ({'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic},
            {'Unlearn Truth Ratio': unlearn_truth_ratio, 'Retain Truth Ratio': retain_truth_ratio}) 

def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def length_auroc(rouge_ls):
    ...
