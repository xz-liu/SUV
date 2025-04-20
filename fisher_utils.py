#     Computes a differential Fisher regularization loss.
#     For each targeted parameter, we compute:
#
#         diff_importance = max(0, F_forbidden - F_retain + fisher_eps)
#
#     and then penalize the squared magnitude of the parameter (or its effective update)
#     relative to this importance. Here we use a log penalty:
#
#         penalty = log(1 + (param^2 / (diff_importance + fisher_eps)))

import os
import torch
from tqdm import tqdm, trange
import math
from torch.func import functional_call, grad, vmap  # requires PyTorch 2.x with torch.func
import torch.nn.functional as F


##########################################
# HELPER: COMPUTE FISHER INFORMATION FOR BASE PARAMETERS (Memory-Optimized)
##########################################
def compute_fisher_base(model, dataloader, device, eps=1e-6, dtype=torch.float16, use_cache=True, model_name=None,
                        fisher_type=None):
    """
    Computes an estimate of the Fisher Information for each base parameter in target modules.
    To save GPU memory, all Fisher tensors are allocated on the CPU.

    Parameters:
      - model: the model (with LoRA applied) but used here with LoRA bypassed/disabled for the base params.
      - dataloader: forbidden dataset loader
      - device: training device (e.g., cuda)
      - eps: a small epsilon to avoid division by zero
      - dtype: data type for storing Fisher info (can be torch.float16 to further reduce memory usage)
    Returns:
      - fisher: a dictionary mapping base parameter names to Fisher info tensors stored on the CPU.
    """
    if use_cache:
        assert model_name is not None and fisher_type is not None
        disk_cache = DiskCacheWrapper(model_name, fisher_type)
        # TODO: REMOVE INVALIDATE CACHE
        disk_cache.invalidate_cache()
        fisher = disk_cache.load_fisher()
        if fisher is not None:
            return fisher
    else:
        disk_cache = None

    model.eval()
    fisher = {}
    # Define target modules (the ones to which LoRA is applied)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Allocate Fisher tensors on CPU (using the given dtype) for all target base parameters.
    for name, param in tqdm(model.named_parameters(), desc="Init Fisher Information"):
        if any(tm in name for tm in target_modules) and ("lora_" not in name):
            # Allocate on CPU to save GPU memory.
            fisher[name] = torch.zeros_like(param.data, device=param.device, dtype=dtype,requires_grad=False)
            # print('Added', name, 'to Fisher Information')

    # input()
    # Temporarily ensure gradients are enabled for these parameters.
    original_requires_grad = {}
    for name, param in model.named_parameters():
        if name in fisher:
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

    # Loop over the forbidden dataset
    i = 0
    for batch in tqdm(dataloader, desc="Computing Fisher Information"):
        model.zero_grad()
        # Move the batch to the training device.
        batch = {k: v.to(device) for k, v in batch.items()}
        # if Batch is 1D tensor, add a batch dimension
        for k, v in batch.items():
            if len(v.shape) == 1:
                batch[k] = v.unsqueeze(0)
        # print('Batch:', batch)
        print('fuck',i)
        if i ==141:
            print('fuck')

        i+=1
        outputs = model(**batch, labels=batch["input_ids"])
        params = [p for p in model.parameters() if p.requires_grad]
        assert len(params) > 0, "No parameters to compute Fisher information for."
        grads = torch.autograd.grad(outputs.loss, params, create_graph=False)

        # print("Shape of grads: ", grads.shape)
        # Then, if you need to update your fisher dictionary only for parameters of interest:
        param_names = [name for name, param in model.named_parameters() if param.requires_grad]
        for name, grad in zip(param_names, grads):
            if name in fisher and grad is not None:
                fisher[name] += grad.detach().to(fisher[name].device).to(dtype).pow(2).sum(dim=0)

    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad = original_requires_grad[name]
    # Average over the number of batches and log summary statistics.
    for name in fisher:
        fisher[name] /= len(dataloader)
        print(
            f"Fisher base - {name}: mean {fisher[name].mean().item()}, max {fisher[name].max().item()}, min {fisher[name].min().item()}")
        fisher[name] = fisher[name].to('cpu')  # move to cpu

    if use_cache:
        disk_cache.save_fisher(fisher)
    return fisher


import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap  # requires PyTorch 2.x
from tqdm import tqdm


def compute_fisher_per_gradient(model, dataloader, device, eps=1e-6, dtype=torch.float16,
                                use_cache=True, model_name=None, fisher_type=None):
    """
    Computes an estimate of the Fisher Information for each base parameter in target modules.
    This version uses function transforms to compute per-sample gradients in a batched (vectorized)
    manner using the original model parameters. To save GPU memory, all Fisher tensors are allocated
    on the CPU.

    Parameters:
      - model: the model (with LoRA applied) but with LoRA bypassed for the base params.
      - dataloader: forbidden dataset loader
      - device: training device (e.g., 'cuda')
      - eps: a small epsilon to avoid division by zero
      - dtype: data type for storing Fisher info (e.g., torch.float16 for reduced memory usage)

    Returns:
      - fisher: a dictionary mapping base parameter names to Fisher info tensors stored on the CPU.
    """
    # --- Cache handling (if using cache) ---
    if use_cache:
        assert model_name is not None and fisher_type is not None
        disk_cache = DiskCacheWrapper(model_name, fisher_type)
        disk_cache.invalidate_cache()  # TODO: REMOVE INVALIDATE CACHE
        fisher = disk_cache.load_fisher()
        if fisher is not None:
            return fisher
    else:
        disk_cache = None

    model.eval()
    fisher = {}
    # Define target modules (the ones to which LoRA is applied)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # Allocate Fisher tensors on CPU (using the given dtype) for all target base parameters.
    for name, param in tqdm(model.named_parameters(), desc="Init Fisher Information"):
        if any(tm in name for tm in target_modules) and ("lora_" not in name):
            fisher[name] = torch.zeros_like(param.data, device=param.device, dtype=dtype, requires_grad=False)

    # Temporarily ensure gradients are enabled for these parameters.
    original_requires_grad = {}
    for name, param in model.named_parameters():
        if name in fisher:
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

    # Define the original loss function which uses the model's built-in loss.
    def compute_loss(func_params, func_buffers, sample, target):
        sample = sample.unsqueeze(0)  # add batch dimension
        target = target.unsqueeze(0)
        # Optionally print shapes for debugging.
        print("Shape of the sample: ", sample.shape)
        print("Shape of the target: ", target.shape)
        # Pass labels so that the model computes its own loss.
        outputs = functional_call(model, (func_params, func_buffers), (sample,), dict(labels=target))
        return outputs.loss

    # Define a helper function that computes and prunes gradients for a single sample.
    def compute_pruned_grad(func_params, func_buffers, sample, target):
        # Compute gradients with respect to the loss.
        grads = grad(compute_loss)(func_params, func_buffers, sample, target)
        # Prune the result: keep only those items that exist in fisher.
        pruned_grads = {k: v for k, v in grads.items() if k in fisher}
        return pruned_grads

    # Vectorize the gradient computation (with pruning) over the batch dimension.
    ft_compute_sample_grad = vmap(compute_pruned_grad, in_dims=(None, None, 0, 0))

    # Loop over the forbidden dataset.
    for batch in tqdm(dataloader, desc="Computing Fisher Information"):
        model.zero_grad()  # Zero out gradients in the model for each iteration.
        input_ids = batch["input_ids"].to(device)
        # In this example, we assume the inputs serve as both inputs and targets.

        # Use the original parameters for the functional call.
        func_params = {k: v for k, v in dict(model.named_parameters()).items() if k in fisher}
        func_buffers = {k: v for k, v in dict(model.named_buffers()).items()}

        # Compute per-sample gradients (pruned within the vmap function).
        per_sample_grads = ft_compute_sample_grad(func_params, func_buffers, input_ids, input_ids)
        # Each entry in per_sample_grads is of shape: (batch_size, *parameter.shape)
        print('fuck')
        # Accumulate the squared per-sample gradients into the Fisher estimates.
        for name, grad_tensor in per_sample_grads.items():
            if grad_tensor is not None:
                fisher[name] += grad_tensor.detach().to(fisher[name].device).to(dtype).pow(2).sum(dim=0)

    # Restore original requires_grad states.
    for name, param in model.named_parameters():
        if name in original_requires_grad:
            param.requires_grad = original_requires_grad[name]

    # Average the accumulated Fisher information over the number of batches and log stats.
    num_batches = len(dataloader)
    for name in fisher:
        fisher[name] /= num_batches
        print(f"Fisher base - {name}: mean {fisher[name].mean().item()}, "
              f"max {fisher[name].max().item()}, min {fisher[name].min().item()}")
        fisher[name] = fisher[name].to('cpu')  # Optionally move to CPU

    if use_cache:
        disk_cache.save_fisher(fisher)
    return fisher


class DiskCacheWrapper:
    def __init__(self, model_name, fisher_type):
        # Check if the Fisher information has been computed and saved to disk
        fisher_cache_path = f"fisher_cache/{model_name}_{fisher_type}_fisher.pt"
        # create the cache directory if it doesn't exist
        # dir for fisher cache path
        dir = os.path.dirname(fisher_cache_path)

        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        self.fisher_cache_path = fisher_cache_path

    def invalidate_cache(self):
        if os.path.exists(self.fisher_cache_path):
            os.remove(self.fisher_cache_path)
            print(f"Fisher information cache at {self.fisher_cache_path} invalidated.")

    def save_fisher(self, fisher):
        torch.save(fisher, self.fisher_cache_path)
        print(f"Fisher information saved to {self.fisher_cache_path}")

    def load_fisher(self):
        if os.path.exists(self.fisher_cache_path):
            fisher = torch.load(self.fisher_cache_path)
            print(f"Fisher information loaded from {self.fisher_cache_path}")
            return fisher
        else:
            print(f"No Fisher information found at {self.fisher_cache_path}")
            return None


def compute_differential_fisher(fisher_forbidden, fisher_retain, fisher_eps=1e-6, scale=1.0):
    differential_fisher = {}
    for name in fisher_forbidden:
        if name in fisher_retain:
            diff_imp = torch.clamp(fisher_forbidden[name] * scale - fisher_retain[name] + fisher_eps, min=fisher_eps)
            differential_fisher[name] = diff_imp
            print(
                f"Differential Fisher - {name}: mean {diff_imp.mean().item()}, max {diff_imp.max().item()}, l2 {diff_imp.norm().item()}")
    return differential_fisher


class NoScheduledLambdaFisher:
    def __init__(self, original_value):
        self.original_value = original_value
        self.current_value = original_value

    def update_loss(self, loss):
        pass

    def __call__(self):
        return self.current_value

    def set_value(self, value):
        self.current_value = value

    def set_zero(self):
        self.current_value = 0.0


class ScheduledLambdaFisher:
    def __init__(self, original_value, max_loss_not_updated_rounds=5, printer=print):
        self.original_value = original_value
        self.current_value = original_value
        self.loss = 0.0
        self.min_loss = float('inf')
        self.max_loss_not_updated_rounds = max_loss_not_updated_rounds
        self.loss_not_updated_rounds = 0
        self.printer = printer

    def update_loss(self, loss):

        self.loss = loss
        if loss < self.min_loss:
            self.min_loss = loss
        else:
            self.loss_not_updated_rounds += 1
            if self.loss_not_updated_rounds >= self.max_loss_not_updated_rounds:
                self.loss_not_updated_rounds = 0
                self.set_divide(10.0)

    def __call__(self):
        return self.current_value

    def set_value(self, value):
        self.current_value = value

    def set_divide(self, value):
        self.current_value = self.current_value / value
        self.printer(f"Lambda set to {self.current_value}, divided by {value}")

    def set_zero(self):
        self.current_value = 0.0


class ScheduledLambdaFisherAdaptive:
    """
    When loss is decreasing, apply mild decay.
    When loss is not decreasing for max_loss_not_updated_rounds, apply severe decay.
    """

    def __init__(self, original_value, mild_decay_rate=0.99, severe_decay_rate=0.9,
                 max_loss_not_updated_rounds=5, printer=print):
        self.original_value = original_value
        self.current_value = original_value
        self.mild_decay_rate = mild_decay_rate
        self.severe_decay_rate = severe_decay_rate
        self.max_loss_not_updated_rounds = max_loss_not_updated_rounds
        self.loss_not_updated_rounds = 0
        self.min_loss = float('inf')
        self.loss = float('inf')
        self.printer = printer

    def update_loss(self, new_loss):
        if new_loss < self.min_loss:
            self.min_loss = new_loss
            self.loss_not_updated_rounds = 0
            self.apply_mild_decay()
        else:
            self.loss_not_updated_rounds += 1
            if self.loss_not_updated_rounds >= self.max_loss_not_updated_rounds:
                self.apply_severe_decay()
                self.loss_not_updated_rounds = 0  # 重置计数器

        self.loss = new_loss

    def apply_mild_decay(self):
        self.current_value *= self.mild_decay_rate

    def apply_severe_decay(self):
        self.current_value *= self.severe_decay_rate

    def __call__(self):
        return self.current_value

    def set_value(self, value):
        self.current_value = value

    def set_zero(self):
        self.current_value = 0.0
