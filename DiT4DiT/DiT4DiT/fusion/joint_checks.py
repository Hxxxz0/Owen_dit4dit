"""Optimizer coverage and independently auditable backbone gradient routes."""
import torch


def optimizer_groups(model, learning_rates):
    groups, seen = [], set()
    for key, lr in learning_rates.items():
        params = [p for p in getattr(model, key).parameters() if p.requires_grad]
        if not params:
            raise ValueError(f"Empty trainable optimizer group: {key}")
        for p in params:
            if id(p) in seen:
                raise ValueError("A parameter appears in multiple optimizer groups")
            seen.add(id(p))
        groups.append({"params": params, "lr": float(lr), "name": key})
    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    if missing:
        raise ValueError(f"Trainable parameters missing from optimizer: {missing[:10]}")
    return groups


def update_anchors(model):
    prefixes = ["cosmos", "fusion", "action_model"]
    if hasattr(model, "qwen"):
        prefixes += ["qwen.visual", "qwen.language_model"]
    anchors = {}
    for prefix in prefixes:
        candidates = [(n, p) for n, p in model.named_parameters()
                      if n.startswith(prefix + ".") and p.ndim >= 2 and p.requires_grad]
        preferred = ("to_q.weight", "self_attn.q_proj.weight", "attn.qkv.weight")
        name, param = next(((n, p) for n, p in candidates if any(x in n for x in preferred)), candidates[0])
        anchors[prefix] = (name, param, param.detach().flatten()[:65536].clone())
    return anchors


def action_gradient_evidence(model, samples):
    """Run before DeepSpeed hooks are installed; no optimizer or weight mutation."""
    anchors = update_anchors(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses = model(samples)
    grads = torch.autograd.grad(losses["action_loss"], [p for _, p, _ in anchors.values()], allow_unused=True)
    evidence = {}
    for (group, (name, _, _)), grad in zip(anchors.items(), grads):
        if grad is None or not torch.isfinite(grad).all() or not grad.count_nonzero():
            raise AssertionError(f"Action loss does not reach {name} with finite nonzero gradients")
        evidence[group] = {"parameter": name, "gradient_norm": float(grad.float().norm()),
                           "gradient_max": float(grad.float().abs().max())}
    return {"action_loss": float(losses["action_loss"].detach()), "action_only_gradients": evidence}


class FirstBackwardAudit:
    """Observe autograd before ZeRO clears gradients; remove hooks after step one."""
    def __init__(self, model):
        self.norms, self.handles, self.expected = {}, [], []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.expected.append(name)
                def record(gradient, key=name):
                    self.norms[key] = gradient.detach().float().norm()
                self.handles.append(param.register_hook(record))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        missing = sorted(set(self.expected)-set(self.norms))
        names = sorted(self.norms)
        values = torch.stack([self.norms[n] for n in names]).cpu().tolist()
        import math
        if missing or not all(math.isfinite(v) for v in values):
            raise AssertionError(f"Full-parameter backward failed: missing={missing}, nonfinite gradients={not all(math.isfinite(v) for v in values)}")
        result = {"trainable_tensors": len(self.expected), "reached_tensors": len(names),
            "zero_gradient_tensors": [n for n, v in zip(names, values) if v == 0],
            "all_finite": True}
        self.norms.clear()
        return result
