# SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning

This repository contains the implementation of SUV (Selective Unlearning for Verbatim data), a framework designed to prevent Large Language Models (LLMs) from memorizing copyrighted content while preserving their overall utility.

Paper: [SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning](https://arxiv.org/abs/2503.22948)

## Overview

SUV introduces a selective unlearning framework that:
- Prevents LLMs from memorizing copyrighted content
- Preserves model utility on unrelated tasks
- Uses Direct Preference Optimization (DPO) with gradient projection and Fisher information regularization
- Validated on a large-scale dataset of 500 famous books

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd release_version
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `continue_books_sentences.py`: Script for processing book sentences
- `fast_evaluate.py`: Fast evaluation utilities
- `fast_sequential_eval.py`: Sequential evaluation implementation
- `fisher_utils.py`: Fisher information utilities
- `new_fisher_dpo_train.py`: DPO training with Fisher regularization
- `new_fisher_project_dpo_train.py`: Projected DPO training with Fisher regularization
- `new_project_dpo_train.py`: Projected DPO training implementation
- `prepare_dpo_dataset.py`: Dataset preparation for DPO
- `sliding_window_generate.py`: Sliding window generation utilities
- `task_vector_merge.py`: Task vector merging utilities
- `utils.py`: General utility functions

## Usage

### Data Preparation

Before training, you need to prepare the dataset:

1. First, generate sliding window samples:
```bash
python sliding_window_generate.py \
    --model_name [base_model_name] \
    --tokenizer_name [tokenizer_name] \
    --book [book_name] \
    --book_folder [path_to_book_folder] \
    --part [part_number] \
    --total_parts [total_parts] \
    --num_gpus [number_of_gpus] \
    --quant [quantization_method] \
    --use_hf [use_huggingface] \
    --bitsandbytes [use_bitsandbytes]
```

2. Then, prepare the DPO dataset:
```bash
python prepare_dpo_dataset.py \
    --target_llm [target_model_name] \
    --dpo_data_folder [path_to_data_folder] \
    --generation_llm [generation_model_name] \
    --max_rows [maximum_rows] \
    --rouge_threshold [rouge_threshold] \
    --npo_mode [npo_mode]
```

Our method has two variants:

### 1. SUV-AS (Adaptive Selection)
This variant uses adaptive selection for unlearning. To run SUV-AS:

```bash
python new_fisher_project_dpo_train.py \
    --model_name_or_path [base_model_path] \
    --dataset_path [path_to_dataset] \
    --output_dir [output_directory] \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512 \
    --lambda_fisher 0.1 \
    --fisher_eps 1e-6 \
    --use_fisher_reg True \
    --use_retain_loss False \
    --retain_dataset_name [retain_dataset_name] \
    --forbidden_dataset_name [forbidden_dataset_name] \
    --retain_dataset_text [retain_dataset_text]
```

### 2. SUV-TV (Task Vector)
This variant uses task vector merging for unlearning. To run SUV-TV:

1. First, run the Fisher-regularized DPO training:
```bash
python new_fisher_dpo_train.py \
    --model_name_or_path [base_model_path] \
    --dataset_path [path_to_dataset] \
    --output_dir [fisher_output_dir] \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512 \
    --lambda_fisher 0.1 \
    --fisher_eps 1e-6 \
    --use_fisher_reg True \
    --use_retain_loss False \
    --retain_dataset_name [retain_dataset_name] \
    --forbidden_dataset_name [forbidden_dataset_name] \
    --model_name [model_name]
```

2. Then, run the projected DPO training:
```bash
python new_project_dpo_train.py \
    --model_name_or_path [base_model_path] \
    --dataset_path [path_to_dataset] \
    --output_dir [project_output_dir] \
    --learning_rate 1e-5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512
```

3. Finally, merge the task vectors:
```bash
python task_vector_merge.py \
    --fisher_model_path [fisher_output_dir] \
    --project_model_path [project_output_dir] \
    --output_dir [final_output_dir] \
    --merge_alpha 0.5  # Adjust this parameter to control the merging ratio
```

### Evaluation
To evaluate the unlearned model:
```bash
python fast_evaluate.py \
    --model_path [path_to_unlearned_model] \
    --test_dataset_path [path_to_test_dataset] \
    --output_dir [evaluation_output_dir]
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{xu2025suv,
  title={SUV: Scalable Large Language Model Copyright Compliance with Regularized Selective Unlearning},
  author={Xu, Tianyang and Liu, Xiaoze and Wu, Feijie and Wang, Xiaoqian and Gao, Jing},
  journal={arXiv preprint arXiv:2503.22948},
  year={2025}
}
```

## License

This project is released under the MIT License. See the LICENSE file for details.

## Contact

For questions about the code or paper, please contact the authors.

## Dataset Structure

### Books Dataset
The project expects a folder containing book text files. Configure the folder location using:

```bash
--books_dataset_folder [path_to_books_folder]  # Default: 'all_books_data_folder'
```

Each book should be in a separate TXT file within this folder. For example:

```
all_books_data_folder/
├── a_tale_of_two_cities.txt
├── pride_and_prejudice.txt
├── moby_dick.txt
└── ... other books ...
```

Example content of `a_tale_of_two_cities.txt`:
```text
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair...
```

Note: Each TXT file should contain the complete text of one book, with proper formatting and minimal preprocessing (e.g., basic newline handling).

### Script-specific Parameters

1. In `prepare_dpo_dataset.py`:
```bash
python prepare_dpo_dataset.py \
    --books_dataset_folder [path_to_books_folder] \
    ... other arguments ...
```

2. In `fast_sequential_eval.py`:
```bash
python fast_sequential_eval.py \
    --books_dataset_folder [path_to_books_folder] \
    ... other arguments ...
```

The scripts will look for books in the specified folder and process them accordingly. Results and intermediate files will be stored in corresponding subdirectories under `results_NEW/[books_dataset_folder]` and `results/[books_dataset_folder]`.