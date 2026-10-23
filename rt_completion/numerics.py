"""Precision selection and non-finite-safe updates for offline RT training.

BF16 has the exponent range of FP32 and must not inherit the unbounded loss
scale used by older checkpoints. FP16 keeps GradScaler's normal overflow
recovery instead of raising before ``step``/``update`` can perform it.
"""

import math

import torch


def _native_cuda_bfloat16(device):
    # Query the selected device, not whichever CUDA device happens to be current.
    # Do not probe support by allocating a tensor or creating a CUDA context here.
    major, _ = torch.cuda.get_device_capability(device)
    cuda_version = torch.version.cuda
    return major >= 8 and cuda_version is not None and int(cuda_version.split(".")[0]) >= 11


def precision_policy(config, device):
    """Return an explicit autocast/scaler policy; CPU ``auto`` stays FP32."""
    device = torch.device(device)
    settings = config.get("train", {})
    requested = str(settings.get("amp_dtype", "auto")).lower()
    if requested not in ("auto", "bfloat16", "float16", "float32"):
        raise ValueError("train.amp_dtype must be auto, bfloat16, float16 or float32")
    if not settings.get("amp", False) or requested == "float32":
        return {"enabled": False, "dtype": torch.float32, "scaling": False, "name": "float32"}
    if device.type == "cpu":
        if requested == "float16":
            raise ValueError("CPU FP16 training is not supported; use float32 or bfloat16")
        name = "bfloat16" if requested == "bfloat16" else "float32"
    elif device.type == "cuda":
        supported = _native_cuda_bfloat16(device)
        if requested == "bfloat16" and not supported:
            raise ValueError("The selected CUDA device does not support native bfloat16")
        name = ("bfloat16" if supported else "float16") if requested == "auto" else requested
    else:
        if requested != "auto":
            raise ValueError(f"Explicit AMP is not supported for device type {device.type}")
        name = "float32"
    return {"enabled": name != "float32", "dtype": getattr(torch, name),
            "scaling": name == "float16", "name": name}


def make_grad_scaler(policy, device):
    """Only FP16 autocast uses dynamic loss scaling."""
    enabled = bool(policy["enabled"] and policy["scaling"] and policy["dtype"] == torch.float16)
    return torch.amp.GradScaler(torch.device(device).type, enabled=enabled)


def graceful_load_scaler(scaler, state):
    """Restore FP16 state, ignoring obsolete/empty BF16 scaler state.

    Returns whether a state was loaded so callers may report migrations.
    """
    if not scaler.is_enabled() or not state:
        return False
    scaler.load_state_dict(state)
    return True


def _gradient_values(gradient):
    return gradient.coalesce().values() if gradient.is_sparse else gradient


def _wide_norm(values):
    # FP64 accumulation avoids FP32 sum-of-squares overflow for finite gradients
    # such as 1e20. FP64 inputs additionally need scaling before squaring.
    if values.dtype in (torch.float64, torch.complex128):
        maximum = values.abs().amax() if values.numel() else values.real.new_zeros(())
        denominator = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
        return torch.linalg.vector_norm(values / denominator) * maximum
    dtype = torch.complex128 if values.is_complex() else torch.float64
    return torch.linalg.vector_norm(values, dtype=dtype)


def _gradient_norms(named_gradients):
    # One host transfer per device rather than a synchronization per parameter.
    groups = {}
    for name, parameter in named_gradients:
        gradient = _gradient_values(parameter.grad.detach())
        groups.setdefault(gradient.device, []).append(
            (name, _wide_norm(gradient), torch.isfinite(gradient).all()))
    norms = []
    invalid = []
    for group in groups.values():
        values = torch.stack([torch.stack((norm, finite.to(torch.float64)))
                              for _, norm, finite in group]).cpu().tolist()
        for (name, _, _), (norm, finite) in zip(group, values):
            if not finite:
                invalid.append(name)
            else:
                norms.append(norm)
    return (None if invalid else math.hypot(*norms)), invalid


@torch.no_grad()
def safe_optimizer_step(model, optimizer, scaler, grad_clip):
    """Unscale, inspect, clip and update without applying non-finite gradients.

    The caller has already executed ``scaler.scale(loss).backward()``. A failed
    update clears gradients and never advances optimizer state. All returned
    numbers are JSON-safe; an undefined gradient norm is represented by None.
    """
    if not math.isfinite(float(grad_clip)) or float(grad_clip) <= 0:
        raise ValueError("train.grad_clip must be finite and strictly positive")
    scaling = scaler is not None and scaler.is_enabled()
    before = float(scaler.get_scale()) if scaling else 1.0
    result = {"updated": False, "grad_norm": 0.0, "nonfinite_parameters": [],
              "overflow": False, "scale_before": before, "scale_after": before,
              "reason": "no_gradients"}
    named_gradients = [(name, parameter) for name, parameter in model.named_parameters()
                       if parameter.grad is not None]
    if not named_gradients:
        optimizer.zero_grad(set_to_none=True)
        return result
    if scaling:
        # This records GradScaler's overflow flags before any clipping changes.
        scaler.unscale_(optimizer)
    total_norm, invalid = _gradient_norms(named_gradients)
    result.update(grad_norm=total_norm, nonfinite_parameters=invalid)
    if invalid:
        if scaling:
            # The scaler owns skip/backoff; do not call optimizer.step directly.
            scaler.step(optimizer)
            scaler.update()
            result.update(overflow=True, scale_after=float(scaler.get_scale()), reason="amp_overflow")
        else:
            result["reason"] = "nonfinite_gradient"
        optimizer.zero_grad(set_to_none=True)
        return result
    if not math.isfinite(total_norm):
        # Only possible if even the true FP64 global norm is unrepresentable.
        # Never multiply finite gradients by NaN/Inf or perform an unsafe step.
        result.update(grad_norm=None, reason="nonfinite_global_norm")
        if scaling:
            scaler.update(new_scale=before)
        optimizer.zero_grad(set_to_none=True)
        return result
    coefficient = min(1.0, float(grad_clip) / (total_norm + 1e-6))
    if coefficient < 1.0:
        for _, parameter in named_gradients:
            if parameter.grad.is_sparse:
                parameter.grad._values().mul_(coefficient)
            else:
                parameter.grad.mul_(coefficient)
    if scaling:
        scaler.step(optimizer)
        scaler.update()
        result["scale_after"] = float(scaler.get_scale())
    else:
        optimizer.step()
    result.update(updated=True, reason="updated")
    return result
