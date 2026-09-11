"""Resident frame payload ownership, independent of model and output history.

Lists keep global IDs using None tombstones. Small pose/focal histories and
diagnostic records are not part of this payload bound.
"""

import numpy as np


FRAME_STORAGE_SCHEMA = "vmem_resident_frames_v1"
PAYLOAD_FIELDS = ("pil_frames", "latents", "encoder_embeddings", "Ks", "surfel_depths")
ARRAY_FIELDS = PAYLOAD_FIELDS[1:]


def owned_frame_array(value, mode):
    return np.asarray(value).copy() if mode == "resident" else value


def payload_summary(pipeline):
    indices, logical_bytes = {}, {}
    owns_storage = True
    for name in PAYLOAD_FIELDS:
        values = getattr(pipeline, name, [])
        indices[name] = [index for index, value in enumerate(values) if value is not None]
        size = 0
        for index in indices[name]:
            value = values[index]
            if name == "pil_frames":
                # The generation path stores RGB PIL images, not encoded PNG bytes.
                size += value.width * value.height * len(value.getbands())
            else:
                size += int(value.nbytes)
                owns_storage = owns_storage and bool(value.flags.owndata)
        logical_bytes[name] = size
    return {
        "schema": FRAME_STORAGE_SCHEMA,
        "mode": getattr(pipeline, "frame_storage", "legacy"),
        "resident_indices": indices,
        "resident_counts": {name: len(ids) for name, ids in indices.items()},
        "logical_bytes": logical_bytes,
        "total_logical_bytes": sum(logical_bytes.values()),
        "arrays_own_storage": owns_storage,
        "durable_frames": getattr(pipeline, "durable_frame_count", 0),
        "pending_evictions": list(getattr(pipeline, "_pending_payload_evictions", [])),
    }


def validate_resident_payloads(pipeline):
    summary = payload_summary(pipeline)
    if summary["mode"] != "resident":
        return summary
    allowed = set(pipeline.get_allowed_memory_indices())
    if pipeline.memory_policy != "unbounded" and len(allowed) > pipeline.memory_budget:
        raise ValueError("Resident bank exceeds the frame budget")
    for name, ids in summary["resident_indices"].items():
        # There is no reconstructed depth before the first action.
        expected = set() if name == "surfel_depths" and not pipeline.surfel_depths else allowed
        if set(ids) != expected:
            raise ValueError(f"Resident {name} does not match the eligible bank")
    if not summary["arrays_own_storage"]:
        raise ValueError("A resident array retains shared backing storage")
    cache = getattr(pipeline, "_dino_feature_cache", {})
    if not set(cache).issubset(allowed) or any(not value.flags.owndata for value in cache.values()):
        raise ValueError("Feature cache retains evicted payloads or shared backing storage")
    if summary["pending_evictions"]:
        raise ValueError("Payload eviction has not been committed")
    if summary["durable_frames"] != len(pipeline.pil_frames):
        raise ValueError("Resident payloads require durable output for every frame")
    return summary


def release_evicted_payloads(pipeline, durable_frame_count):
    """Commit only after output is durable and Navigator/temporary aliases are gone."""
    if getattr(pipeline, "frame_storage", "legacy") != "resident":
        return
    if durable_frame_count != len(pipeline.pil_frames):
        raise ValueError("Cannot release payloads before all generated frames are durable")
    evicted = list(pipeline._pending_payload_evictions)
    allowed = set(pipeline.get_allowed_memory_indices())
    if any(index in allowed for index in evicted):
        raise ValueError("Refusing to release an eligible frame")
    for index in evicted:
        for name in PAYLOAD_FIELDS:
            values = getattr(pipeline, name)
            if index < len(values):
                values[index] = None
        getattr(pipeline, "_dino_feature_cache", {}).pop(index, None)
    pipeline._pending_payload_evictions = []
    pipeline.durable_frame_count = durable_frame_count
    validate_resident_payloads(pipeline)
