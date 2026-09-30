"""Narrow, explicit compatibility rules for repartitioning an effective batch."""
import copy


def validate_resume_config(saved, current, allow_batch_repartition=False):
    before, after = copy.deepcopy(saved), copy.deepcopy(current)
    old_batch = int(before["datasets"]["vla_data"]["per_device_batch_size"])
    new_batch = int(after["datasets"]["vla_data"]["per_device_batch_size"])
    old_accum = int(before["trainer"]["gradient_accumulation_steps"])
    new_accum = int(after["trainer"]["gradient_accumulation_steps"])
    changed = (old_batch, old_accum) != (new_batch, new_accum)
    if min(old_batch, new_batch, old_accum, new_accum) < 1:
        raise ValueError("Batch size and accumulation must be positive")
    if changed and allow_batch_repartition:
        if old_batch * old_accum != new_batch * new_accum:
            raise ValueError("Resume repartition must preserve the effective batch")
        after["datasets"]["vla_data"]["per_device_batch_size"] = old_batch
        after["trainer"]["gradient_accumulation_steps"] = old_accum
    for key in ("framework", "trainer", "datasets", "seed"):
        if before[key] != after[key]:
            raise ValueError(f"Resume configuration changed: {key}")
    return {"changed": changed, "old_batch": old_batch, "new_batch": new_batch,
            "old_accumulation": old_accum, "new_accumulation": new_accum}
