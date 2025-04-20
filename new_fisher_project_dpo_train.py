import os
import json
import torch
from datasets import Dataset, load_dataset
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, TaskType
import math
import random
from torch.utils.data import Sampler, Dataset, DataLoader
from accelerate import Accelerator
import argparse
from mem_rate_prediction.mem_pred import LLMClassifierMLP
from safetensors.torch import load_file
from tqdm import tqdm, trange

# Initialize Accelerator for distributed training / mixed precision.
accelerator = Accelerator()
last_mem_indices = []
from fisher_utils import compute_fisher_base, compute_differential_fisher, DiskCacheWrapper, ScheduledLambdaFisherAdaptive


##########################################
# CALLBACKS
##########################################
class SaveEveryNStepsCallback(TrainerCallback):
    def __init__(self, save_steps, output_dir):
        self.save_steps = save_steps
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            accelerator.save_state(self.output_dir)
            print(f"Saved model at step {state.global_step} to {self.output_dir}")


class LogAdditionalLossesCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            poplist = []
            for key, value in logs.items():
                if 'loss' in key:
                    # logs['train/' + key] = value
                    poplist.append(key)
            for key in poplist:
                value = logs.pop(key)
                logs['train/' + key] = value

            # if "dpo_loss" in logs and "retain_loss" in logs:
            #     logs["train/dpo_loss"] = logs.pop("dpo_loss")
            #     logs["train/retain_loss"] = logs.pop("retain_loss")
            # if "fisher_loss" in logs:
            #     logs["train/fisher_loss"] = logs.pop("fisher_loss")


##########################################
# UTILITY: ALPACA PROMPT BUILDER
##########################################
def build_alpaca_prompt(instruction, input_text):
    if input_text.strip():
        return (
            f"Below is an instruction that describes a task, paired with an input that provides further context.\n"
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n"
        )
    else:
        return (
            f"Below is an instruction that describes a task.\n"
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )


##########################################
# BATCH SAMPLER AND MULTI-TASK DATASET
##########################################
class MultiTaskBatchSampler(Sampler):
    def __init__(self, len_pref, len_retain, batch_size, pref_num, shuffle=True):
        self.len_pref = len_pref
        self.len_retain = len_retain
        self.batch_size = batch_size
        self.pref_num = pref_num
        self.retain_num = batch_size - pref_num
        self.shuffle = shuffle

        self.pref_indices = list(range(len_pref))
        self.retain_indices = list(range(len_retain))
        self.num_batches = math.ceil(min(len_pref, len_retain) / self.pref_num)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.pref_indices)
            random.shuffle(self.retain_indices)
        p_ptr = 0
        r_ptr = 0
        for _ in range(self.num_batches):
            batch_pref_idx = self.pref_indices[p_ptr: p_ptr + self.pref_num]
            p_ptr += self.pref_num
            if len(batch_pref_idx) < self.pref_num and self.pref_num - len(batch_pref_idx) <= len(self.pref_indices):
                batch_pref_idx += random.sample(self.pref_indices, self.pref_num - len(batch_pref_idx))
            batch_retain_idx = self.retain_indices[r_ptr: r_ptr + min(self.retain_num, len(batch_pref_idx))]
            r_ptr += self.retain_num
            if len(batch_retain_idx) < self.retain_num:
                batch_retain_idx += random.sample(self.retain_indices, self.retain_num - len(batch_retain_idx))
            batch_indices = [(idx, 'pref') for idx in batch_pref_idx] + [(idx, 'retain') for idx in batch_retain_idx]
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


class MultiTaskDataset(Dataset):
    def __init__(self, pref_dataset, retain_dataset):
        self.pref_dataset = pref_dataset
        self.retain_dataset = retain_dataset

    def __getitem__(self, index_and_flag):
        idx, dataset_flag = index_and_flag
        if dataset_flag == 'pref':
            return self.pref_dataset[idx]
        elif dataset_flag == 'retain':
            return self.retain_dataset[idx]
        else:
            raise ValueError("Unknown dataset flag.")

    def __len__(self):
        return len(self.pref_dataset) + len(self.retain_dataset)


##########################################
# NEW: FORBIDDEN DATASET
##########################################
class ForbiddenDataset(Dataset):
    """
    This dataset is created from the preference dataset by concatenating the 'prompt' and 'rejected'
    fields of each sample. The resulting text is tokenized and returned in a format compatible with
    compute_fisher_base (i.e. a dict with key "input_ids").
    """

    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Concatenate the 'prompt' and 'rejected' fields.
        text = item["prompt"] + " " + item["rejected"]
        tokenized = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        # Remove the batch dimension.
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
        return tokenized


class RetainAlpacaDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512, max_samples=256):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        assert len(self.dataset) >= self.max_samples
        self.random_samples = random.sample(range(len(self.dataset)), self.max_samples)

    def __len__(self):
        return min(len(self.dataset), self.max_samples)

    def __getitem__(self, idx):
        item = self.dataset[self.random_samples[idx]]
        # item = self.dataset[idx]
        prompt_str = build_alpaca_prompt(item["instruction"], item["input"])
        full_text = prompt_str + item["output"]
        tokenized = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512, padding=False)
        tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

        return tokenized


##########################################
# CUSTOM TRAINER: MY DPO TRAINER WITH FISHER REGULARIZATION (LoRA-specific, Option 2)
##########################################
class MyDPOTrainer(DPOTrainer):
    def __init__(self, custom_train_loader=None, forbidden_dataloader=None, lambda_fisher=0.1, fisher_eps=1e-6,
                 use_fisher_reg=True, use_retain_loss=False, retain_dataloader=None,
                 model_name=None, forbidden_dataset_name=None, retain_dataset_name=None, retain_dataset_text=None,
                 *args, **kwargs):
        """
        In this option we assume that the only updated parameters are the LoRA matrices.
        The Fisher information is computed on the base parameters (with LoRA disabled) and stored on CPU.
        Then, during training, the effective update (lora_B @ lora_A) is regularized using the corresponding
        Fisher info.
        """
        super().__init__(*args, **kwargs)
        self.custom_train_loader = custom_train_loader
        self.forbidden_dataloader = forbidden_dataloader
        self.lambda_fisher = ScheduledLambdaFisherAdaptive(lambda_fisher)
        self.fisher_eps = fisher_eps
        self.use_fisher_reg = use_fisher_reg
        self.use_retain_loss = use_retain_loss
        self.retain_dataloader = retain_dataloader
        self.model_name = model_name
        self.forbidden_dataset_name = forbidden_dataset_name
        self.retain_dataset_name = retain_dataset_name
        self.retain_dataset_text = retain_dataset_text
        self.sliding_preserve_grad = None  # Store EMA updated preserve gradient
        self.ema_alpha = 0.9  # EMA smoothing factor

    def collate_preserve(self, batch):
        # Create a small batch preserve dataloader with small batch size to save memory
        preserve_dataloader = DataLoader(
            self.retain_dataloader.dataset,
            batch_size=2,   # Adjust based on memory
            shuffle=True,
            collate_fn=self.collate_preserve  # Define collate_fn suitable for preserve data
        )
        return batch

    def collect_preserve_grads(self, retain_dataset_text, use_cache=True):
        # Preparation phase: accumulate gradients from multiple preserve batches (e.g., 100 batches)
        # Only for parameters that need gradients
        if self.sliding_preserve_grad is None:
            self.sliding_preserve_grad = {}
            # Only consider parameters that need gradients
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    self.sliding_preserve_grad[name] = torch.zeros_like(p, device='cpu')

        # Calculate gradients without retaining computation graph, only for trainable_params
        for _ in range(100):
            preserve_batch = next(self.retain_dataloader)
            preserve_loss = self.compute_preserve_loss(self.model, preserve_batch)
            preserve_loss.backward()

            for name, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    self.sliding_preserve_grad[name] += p.grad.cpu()
                    p.grad = None

        # Normalize the accumulated gradients
        for name in self.sliding_preserve_grad:
            self.sliding_preserve_grad[name] /= 100

    def compute_preserve_loss(self, model, preserve_batch):
        """
        给定 preserve 数据 batch，构造输入并计算 preserve loss。
        参考 compute_loss 中对 retain_batch 的处理。
        """
        instructions = [item["instruction"] for item in preserve_batch]
        inputs_texts = [item["input"] for item in preserve_batch]
        outputs_texts = [item["output"] for item in preserve_batch]

        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        for instr, inp, outp in zip(instructions, inputs_texts, outputs_texts):
            prompt_str = build_alpaca_prompt(instr, inp)
            full_text = prompt_str + outp
            tokenized = self.processing_class(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            input_ids = tokenized["input_ids"][0]
            attention_mask = tokenized["attention_mask"][0]

            tokenized_prompt = self.processing_class(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False
            )
            prompt_len = tokenized_prompt["input_ids"].shape[-1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.processing_class.pad_token_id
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            all_attention_mask, batch_first=True, padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        padded_input_ids = padded_input_ids.to(self.args.device)
        padded_attention_mask = padded_attention_mask.to(self.args.device)
        padded_labels = padded_labels.to(self.args.device)

        out = model(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            labels=padded_labels
        )
        return out.loss.mean()

    def get_train_dataloader(self):
        if self.custom_train_loader is not None:
            return self.custom_train_loader
        else:
            return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # ============ 1) Parse inputs, extract pref_batch / retain_batch ============
        pref_batch = inputs["pref_batch"]  # For DPO training (prompt, chosen, rejected)
        retain_batch = inputs["retain_batch"]  # For MMLU or other tasks

        # ============ 2) Convert pref_batch to format required by DPOTrainer.compute_loss ============
        # The parent class DPOTrainer's default compute_loss expects values from batch["prompt_ids"] / ["chosen_ids"] / ["rejected_ids"]
        # Therefore, we need to tokenize prompt, chosen, rejected and concatenate them into corresponding fields.

        # 2.1 Collect text
        prompts = [item["prompt"] for item in pref_batch]
        chosen = [item["chosen"] for item in pref_batch]
        rejected = [item["rejected"] for item in pref_batch]

        # 2.2 Tokenize prompt / chosen / rejected separately
        # Note: ensure they are on the correct device (e.g., self.args.device)
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.args.device)
        chosen_ids = self.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.args.device)
        rejected_ids = self.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.args.device)

        # 2.3 Construct input format for parent class DPOTrainer
        # This matches trl 0.4.x's DPOTrainer.compute_loss
        # The concatenation of prompt + chosen / prompt + rejected will be done in the parent class,
        # or the parent class may only need separate tokens, depending on the version.
        # If the parent class requires concatenation, we need to concatenate here;
        # If the parent class only needs separate tokens, we pass them as shown below.

        dpo_inputs = {
            "prompt_ids": prompt_ids["input_ids"],
            "prompt_attention_mask": prompt_ids["attention_mask"],
            "chosen_ids": chosen_ids["input_ids"],
            "chosen_attention_mask": chosen_ids["attention_mask"],
            "rejected_ids": rejected_ids["input_ids"],
            "rejected_attention_mask": rejected_ids["attention_mask"],
        }

        # ============ 3) Call parent class's compute_loss to get dpo_loss ============
        # Note: must set return_outputs to True to get (loss, outputs) return value
        dpo_loss, dpo_outputs = super().compute_loss(model, dpo_inputs, return_outputs=True)

        # We can also return some intermediate values for debugging or logging
        if return_outputs:
            return dpo_loss, dpo_outputs
        return dpo_loss

    def training_step(self, model, inputs, num_items_in_batch):
        # Calculate unlearning loss gradient (only for parameters that need gradients)
        unlearning_loss = self.compute_loss(model, inputs)
        unlearning_loss.backward()

        # Calculate current preserve loss and its gradient
        preserve_batch = next(self.retain_dataloader)
        preserve_loss = self.compute_preserve_loss(model, preserve_batch)
        preserve_loss.backward()

        # Update sliding_preserve_grad with EMA
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if name in self.sliding_preserve_grad:
                    self.sliding_preserve_grad[name] = self.ema_alpha * self.sliding_preserve_grad[name] + \
                                                     (1 - self.ema_alpha) * p.grad.cpu()
                else:
                    self.sliding_preserve_grad[name] = p.grad.cpu()
                p.grad = None

        # Gradient projection: for each parameter to be updated, project unlearning gradient
        # onto the orthogonal direction of sliding preserve gradient
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.sliding_preserve_grad:
                preserve_grad = self.sliding_preserve_grad[name].to(p.device)
                unlearning_grad = p.grad

                # Project unlearning_grad onto preserve_grad
                dot_product = torch.sum(unlearning_grad * preserve_grad)
                preserve_norm_squared = torch.sum(preserve_grad * preserve_grad)
                if preserve_norm_squared > 0:
                    projection = (dot_product / preserve_norm_squared) * preserve_grad
                    combined = unlearning_grad - projection
                else:
                    combined = unlearning_grad

                p.grad = combined  # Write back the projected gradient


##########################################
# MAIN FUNCTION
##########################################
def main():
    parser = argparse.ArgumentParser(description="Train DPO with LoRA on retain and preference datasets.")
    parser.add_argument("--dpo_data_folder", type=str, required=True, help="Path to the folder containing DPO data.")
    parser.add_argument("--dpo_data_file", type=str, default="dpo_dataset.json", help="DPO data file.")
    parser.add_argument("--retain_dataset_name", type=str, default="yahma/alpaca-cleaned", help="Retain dataset name.")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name.")
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str, default="test_output_dir", help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--pref_num", type=int, default=1, help="Number of preference samples per batch.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")

    args = parser.parse_args()

    # Load datasets.
    with open(os.path.join(args.dpo_data_folder, args.dpo_data_file), "r") as f:
        dpo_data = json.load(f)
    dpo_data["index"] = [i for i in range(len(dpo_data["prompt"]))]
    pref_dataset = datasets.Dataset.from_dict(dpo_data)
    retain_dataset = load_dataset(args.retain_dataset_name)["train"]

    # Load base model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # Configure and inject LoRA adapters.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("LoRA parameters injected.")

    # Create the multi-task dataset and sampler.
    multi_task_dataset = MultiTaskDataset(pref_dataset, retain_dataset)
    sampler = MultiTaskBatchSampler(
        len_pref=len(pref_dataset),
        len_retain=min(len(retain_dataset), len(pref_dataset)),
        batch_size=args.batch_size,
        pref_num=args.pref_num,
        shuffle=True
    )

    def collate_fn(batch):
        pref_samples = []
        retain_samples = []
        for item in batch:
            if "prompt" in item and "chosen" in item and "rejected" in item:
                pref_samples.append(item)
            else:
                retain_samples.append(item)
        return {
            "pref_batch": pref_samples,
            "retain_batch": retain_samples
        }

    train_loader = DataLoader(multi_task_dataset, batch_sampler=sampler, collate_fn=collate_fn)
    train_loader = accelerator.prepare(train_loader)

    retain_alpaca_dataset = RetainAlpacaDataset(retain_dataset, tokenizer)
    retain_loader = DataLoader(retain_alpaca_dataset, batch_size=1)

    # Create the forbidden dataset from the preference dataset by concatenating 'prompt' and 'rejected'
    forbidden_dataset = ForbiddenDataset(pref_dataset, tokenizer, max_length=512)
    forbidden_loader = DataLoader(forbidden_dataset, batch_size=1)

    training_args = DPOConfig(
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=1.0,
        bf16=True,
        save_only_model=True,
        # save_steps=len(train_loader),
    )

    dummy_data = {"prompt": [""], "chosen": [""], "rejected": [""]}
    dummy_dataset = datasets.Dataset.from_dict(dummy_data)

    # Initialize our custom trainer with the forbidden_loader passed via forbidden_dataloader.
    trainer = MyDPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=dummy_dataset,
        custom_train_loader=train_loader,
        callbacks=[LogAdditionalLossesCallback, SaveEveryNStepsCallback(500, args.output_dir)],
        forbidden_dataloader=forbidden_loader,
        retain_dataloader=retain_loader,

        # MODELNAMEs,

        model_name=args.base_model,
        forbidden_dataset_name=args.dpo_data_folder + args.dpo_data_file,
        retain_dataset_name=args.retain_dataset_name,
        # lambda_fisher=1e-9,
        retain_dataset_text=retain_dataset,
    )

    trainer.train()
    #
    # curr_epoch = 1
    # global last_mem_indices
    # while curr_epoch < args.num_train_epochs:
    #     print(f"Epoch {curr_epoch} finished. Start next epoch.")
    #     sampler = MultiTaskBatchSampler(
    #         len_pref=len(pref_dataset),
    #         len_retain=len(retain_dataset),
    #         batch_size=args.batch_size,
    #         pref_num=args.pref_num,
    #         shuffle=True
    #     )
    #     multi_task_dataset = MultiTaskDataset(pref_dataset, retain_dataset)
    #     train_loader = DataLoader(multi_task_dataset, batch_sampler=sampler, collate_fn=collate_fn)
    #     train_loader = accelerator.prepare(train_loader)
    #     trainer.custom_train_loader = train_loader
    #     last_mem_indices = []
    #     trainer.train()
    #     curr_epoch += 1

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    print("Done training & saved model.")


if __name__ == "__main__":
    main()
