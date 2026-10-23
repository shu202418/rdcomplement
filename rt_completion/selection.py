"""Shared export operating point; no GT, SMPL-X, or negative-empty labels."""

import torch


@torch.no_grad()
def select_additions(model, raw, prediction, settings):
    """Return the actual, budgeted additions used by export and held-out checks.

    ``probability`` is a legacy field name for an uncalibrated reflection score.
    Original nonzero observations can never be overwritten by this selector.
    """
    probability = prediction["reflection_logits"].sigmoid()
    sigma = (0.5 * prediction["log_variance"]).exp()
    values = probability * model.denormalize(prediction["intensity"])
    candidate = ((raw[:, -1:] == 0)
                 & (probability >= settings["probability_threshold"])
                 & (sigma <= settings["max_sigma_log"])
                 & (values >= settings["min_intensity"])
                 & (prediction["body_prior"] >= settings["body_threshold"]))
    accepted = torch.zeros_like(candidate)
    budget = int(settings["max_added_voxels"])
    if budget < 0:
        raise ValueError("max_added_voxels must be non-negative")
    for batch in range(raw.shape[0]):
        indices = candidate[batch].flatten().nonzero().flatten()
        if len(indices) > budget:
            score = (probability[batch] * values[batch]
                     / sigma[batch].clamp_min(1e-4)).flatten()[indices]
            indices = indices[score.topk(budget).indices]
        accepted[batch].view(-1)[indices] = True
    return {"accepted": accepted, "probability": probability,
            "sigma": sigma, "values": values}
