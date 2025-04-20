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
from functools import partial
from torch.utils.data import Sampler, Dataset, DataLoader
from accelerate import Accelerator
import argparse
from mem_rate_prediction.mem_pred import LLMClassifierMLP
from safetensors.torch import load_file
from tqdm import tqdm, trange

# Initialize Accelerator for distributed training / mixed precision.
accelerator = Accelerator()
last_mem_indices = []
from fisher_utils import compute_fisher_base, compute_differential_fisher, ScheduledLambdaFisher, \
    ScheduledLambdaFisherAdaptive, NoScheduledLambdaFisher


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
            if "dpo_loss" in logs and "retain_loss" in logs:
                logs["train/dpo_loss"] = logs.pop("dpo_loss")
                logs["train/retain_loss"] = logs.pop("retain_loss")
            if "fisher_loss" in logs:
                logs["train/fisher_loss"] = logs.pop("fisher_loss")


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
        return text


class RetainAlpacaDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512, max_samples=4096):
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
        return full_text


def collate_for_fisher(batch, tokenizer):
    tokenized = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512,
                          padding=True)
    return tokenized


##########################################
# CUSTOM TRAINER: MY DPO TRAINER WITH FISHER REGULARIZATION (LoRA-specific, Option 2)
##########################################
class MyDPOTrainer(DPOTrainer):
    def __init__(self, custom_train_loader=None, mem_model=None, mem_threshold=0.4, mem_scale_factor=1e2,
                 forbidden_dataloader=None, lambda_fisher=0.1, fisher_eps=1e-6,
                 use_mem_detect=False, use_fisher_reg=True, use_retain_loss=False, retain_dataloader=None,
                 model_name=None, forbidden_dataset_name=None, retain_dataset_name=None,
                 *args, **kwargs):
        """
        In this option we assume that the only updated parameters are the LoRA matrices.
        The Fisher information is computed on the base parameters (with LoRA disabled) and stored on CPU.
        Then, during training, the effective update (lora_B @ lora_A) is regularized using the corresponding
        Fisher info.
        """
        super().__init__(*args, **kwargs)
        self.custom_train_loader = custom_train_loader
        self.mem_model = mem_model
        self.mem_threshold = mem_threshold
        self.mem_scale_factor = mem_scale_factor
        self.lambda_fisher = NoScheduledLambdaFisher(lambda_fisher)
        self.fisher_eps = fisher_eps

        self.use_mem_detect = use_mem_detect
        self.use_fisher_reg = use_fisher_reg
        self.use_retain_loss = use_retain_loss

        if forbidden_dataloader is not None:
            # Compute Fisher information on the base parameters and store it on CPU.
            self.forbidden_fisher = compute_fisher_base(self.model, forbidden_dataloader, self.args.device,
                                                        eps=fisher_eps, dtype=torch.bfloat16,
                                                        use_cache=model_name is not None,
                                                        model_name=model_name, fisher_type=forbidden_dataset_name)
            if retain_dataloader is not None:
                self.retain_fisher = compute_fisher_base(self.model, retain_dataloader, self.args.device,
                                                         eps=fisher_eps, dtype=torch.bfloat16,
                                                         use_cache=model_name is not None, model_name=model_name,
                                                         fisher_type=retain_dataset_name)
                self.differential_fisher = compute_differential_fisher(self.forbidden_fisher, self.retain_fisher, float(
                    len(retain_dataloader) / len(forbidden_dataloader)))
                self.assign_device_for_fisher(self.model)

            print("Computed Fisher information for base parameters.")
        else:
            self.forbidden_fisher = None

    def get_train_dataloader(self):
        if self.custom_train_loader is not None:
            return self.custom_train_loader
        else:
            return super().get_train_dataloader()

    def trainable_check(self, model):
        if hasattr(self, 'trainable_checked'):
            return
        for name, param in model.named_parameters():
            if 'lora' in name:
                accelerator.print(name, param.requires_grad)
                param.requires_grad = True

        # input()
        self.trainable_checked = True

    def small_kaiming_init_on_lora_B(self, model):
        if hasattr(self, 'init_done'):
            return

        for name, param in model.named_parameters():
            if 'lora_B' in name:
                # the init value should be small
                torch.nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                param.data = param.data * 0.01

        self.init_done = True

    def assign_device_for_fisher(self, model):
        model_params = dict(model.named_parameters())
        accelerator.print('Assigning Fisher information to the device of the corresponding model parameters.')
        # accelerator.print(model_params.keys())
        # accelerator.print(self.differential_fisher.keys())
        for name in self.differential_fisher:
            # model_name = 'module.' + name
            self.differential_fisher[name] = self.differential_fisher[name].to(model_params[name].device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        # self.trainable_check(model)
        # self.small_kaiming_init_on_lora_B(model)
        # --- Add a forward hook to capture hidden states (for memory detection, if used) ---
        captured_logits = []
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            last_layer = model.module.base_model.model.model.layers[-1]
        else:
            last_layer = model.base_model.model.model.layers[-1]

        def hook_fn(module, input, output):
            captured_logits.append(output)

        hook_handle = last_layer.register_forward_hook(hook_fn)

        # ============ 1) Split inputs into pref and retain batches ============
        pref_batch = inputs["pref_batch"]  # For DPO training (prompt, chosen, rejected)
        retain_batch = inputs["retain_batch"]  # For additional tasks (e.g. MMLU)

        # ============ 2) Process the pref_batch for DPO loss ============
        prompts = [item["prompt"] for item in pref_batch]
        chosens = [item["chosen"] for item in pref_batch]
        rejecteds = [item["rejected"] for item in pref_batch]

        tokenized_prompt = self.processing_class(prompts, padding=True, truncation=True, return_tensors="pt").to(
            self.args.device)
        tokenized_chosen = self.processing_class(chosens, padding=True, truncation=True, return_tensors="pt").to(
            self.args.device)
        tokenized_rejected = self.processing_class(rejecteds, padding=True, truncation=True, return_tensors="pt").to(
            self.args.device)

        dpo_inputs = {
            "prompt_input_ids": tokenized_prompt["input_ids"],
            "prompt_attention_mask": tokenized_prompt["attention_mask"],
            "chosen_input_ids": tokenized_chosen["input_ids"],
            "chosen_attention_mask": tokenized_chosen["attention_mask"],
            "rejected_input_ids": tokenized_rejected["input_ids"],
            "rejected_attention_mask": tokenized_rejected["attention_mask"],
        }

        # ============ 3) Get DPO loss from the parent class ============
        dpo_loss, dpo_outputs = super().compute_loss(model=model, inputs=dpo_inputs, return_outputs=True)

        # Remove the hook
        hook_handle.remove()
        batch_loss_scale = 1
        # ============ 4) Compute retain loss (if enabled) ============
        retain_loss = torch.tensor(0.0, device=self.args.device)
        # ============ 5) Compute Fisher regularization loss based on effective LoRA update ============
        fisher_loss = torch.tensor(0.0, device=self.args.device)
        # print('Fisher regularization', 'enabled' if self.use_fisher_reg else 'disabled')
        # print('Forbidden Fisher', 'available' if self.forbidden_fisher is not None else 'unavailable')
        lora_a_accum = 0
        lora_b_accum = 0
        fisher_info_accum = 0
        if self.use_fisher_reg and self.differential_fisher is not None:
            # Create a dictionary for model parameters for quick lookup.
            model_params = dict(model.named_parameters())
            # Iterate over LoRA parameters (we use the ones ending in "lora_B").
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    # Derive the corresponding lora_A parameter name.
                    lora_A_name = name.replace("lora_B", "lora_A")
                    if lora_A_name not in model_params:
                        accelerator.print(f"Warning: {lora_A_name} not found in model parameters.")
                        continue
                    lora_B = param
                    lora_A = model_params[lora_A_name]
                    # Compute effective update (the product should match the update to the base weight).
                    effective_update = torch.matmul(lora_B, lora_A)
                    # Map the LoRA parameter name to the corresponding base parameter name.
                    base_name = name.split(".lora_")[0] + ".base_layer.weight"
                    old_base_name = base_name
                    base_name = base_name.replace("module.", "")
                    if base_name not in self.forbidden_fisher:
                        accelerator.print(f"Warning: {base_name} not found in forbidden Fisher information.")
                        continue
                    # IMPORTANT: Move the Fisher tensor from CPU to the current device.

                    differential_fisher = self.differential_fisher[base_name].to(effective_update.device).to(
                        effective_update.dtype)
                    lora_a_accum += (lora_A ** 2).sum()
                    lora_b_accum += (lora_B ** 2).sum()
                    fisher_info_accum += differential_fisher.sum()

                    #     penalty = log(1 + (param^2 / (diff_importance + fisher_eps)))
                    curr_fisher_loss = 1 + (effective_update ** 2 / (differential_fisher + self.fisher_eps))
                    curr_fisher_loss = torch.log(curr_fisher_loss)
                    # curr_fisher_loss = torch.nn.functional.dropout(curr_fisher_loss, p=0.5, training=True)
                    curr_fisher_loss = curr_fisher_loss.sum()
                    # print(f"Fisher regularization - {name}: {curr_fisher_loss.item()}")
                    fisher_loss += curr_fisher_loss

        # accelerator.print('LORA A:', lora_a_accum)
        # accelerator.print('LORA B:', lora_b_accum)
        # accelerator.print('FISHER INFO:', fisher_info_accum)

        # ============ 6) Combine all losses ============
        total_loss = dpo_loss
        if self.use_mem_detect:
            self.log({"batch_loss_scale": batch_loss_scale})
            total_loss = total_loss * batch_loss_scale
        if self.use_retain_loss:
            total_loss += retain_loss
        if self.use_fisher_reg:
            total_loss += self.lambda_fisher() * torch.clip(fisher_loss, 0, 1e9)

        self.lambda_fisher.update_loss(dpo_loss.detach().item())
        # add a L2 regularization term
        # l2_reg = torch.tensor(0.0, device=self.args.device)
        # for name, param in model.named_parameters():
        #     if 'lora' in name:
        #         l2_reg += torch.norm(param) ** 2
        # total_loss += 1e-7 * l2_reg

        self.log({
            "dpo_loss": dpo_loss.detach().item(),
            "retain_loss": retain_loss.detach().item(),
            "fisher_loss": fisher_loss.detach().item(),
            'log_fisher_loss': torch.log(fisher_loss + 1).detach().item(),
            'clip_log_fisher_loss': torch.clip(torch.log(fisher_loss + 1), 0, 1e9).detach().item(),
            # 'l2_reg': l2_reg.detach().item()

        })

        if return_outputs:
            return (
                total_loss,
                {
                    "dpo_loss": dpo_loss.detach().item(),
                    "retain_loss": retain_loss.detach().item(),
                    "fisher_loss": fisher_loss.detach().item(),
                    "dpo_outputs": dpo_outputs,
                    "batch_loss_scale": batch_loss_scale,
                }
            )
        else:
            return total_loss


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
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--pref_num", type=int, default=1, help="Number of preference samples per batch.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Logging steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--mem_detect_model", type=str, default=None,
                        help="Memory detection model. Set to none to disable.")
    parser.add_argument("--mem_threshold", type=float, default=0.75, help="Memory detection threshold.")
    parser.add_argument("--fisher_batch_size", type=int, default=1, help="Fisher batch size.")

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

    mem_model = None
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
    retain_loader = DataLoader(retain_alpaca_dataset, batch_size=args.fisher_batch_size, shuffle=False,
                               collate_fn=partial(collate_for_fisher, tokenizer=tokenizer))

    # Create the forbidden dataset from the preference dataset by concatenating 'prompt' and 'rejected'
    forbidden_dataset = ForbiddenDataset(pref_dataset, tokenizer, max_length=512)
    forbidden_loader = DataLoader(forbidden_dataset, batch_size=args.fisher_batch_size, shuffle=False, collate_fn=partial(collate_for_fisher, tokenizer=tokenizer))

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
        mem_model=mem_model,
        mem_threshold=args.mem_threshold,
        forbidden_dataloader=forbidden_loader,
        retain_dataloader=retain_loader,

        # MODELNAMEs,

        model_name=args.base_model,
        forbidden_dataset_name=args.dpo_data_folder + args.dpo_data_file,
        retain_dataset_name=args.retain_dataset_name,
        # lambda_fisher=1e-9,
    )

    trainer.train()
    #

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    print("Done training & saved model.")


if __name__ == "__main__":
    main()
