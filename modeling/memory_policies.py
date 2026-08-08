import math
from collections import OrderedDict

import numpy as np


SUPPORTED_MEMORY_POLICIES = (
    "unbounded",
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "mce",
    "kcenter_coreset",
)
BUDGETED_MEMORY_POLICIES = (
    "fifo",
    "rarity_irreplaceability",
    "slam_covisibility",
    "mce",
    "kcenter_coreset",
)


class FrameMemoryBuffer:
    def __init__(self, policy="unbounded", budget=None, pinned_frames=None):
        if policy not in SUPPORTED_MEMORY_POLICIES:
            raise ValueError(
                f"Unsupported memory policy '{policy}'. "
                f"Expected one of {SUPPORTED_MEMORY_POLICIES}."
            )
        if policy in BUDGETED_MEMORY_POLICIES and budget is None:
            raise ValueError(f"{policy} memory policy requires an explicit memory budget")
        if budget is not None and budget <= 0:
            raise ValueError("memory budget must be positive when provided")

        self.policy = policy
        self.budget = budget
        self._frames = OrderedDict()
        self._stats = {}
        self._next_order = 0
        self._pinned_frames = set(int(frame_idx) for frame_idx in (pinned_frames or []))

    def add(self, frame_idx, evict=True, eviction_scores=None, protected_frames=None):
        frame_idx = int(frame_idx)
        if frame_idx not in self._frames:
            self._stats[frame_idx] = {
                "insert_order": self._next_order,
                "selected_count": 0,
                "selection_overlap_sum": 0.0,
                "best_selection_overlap": 0.0,
                "score": 0.0,
            }
            self._next_order += 1
        self._frames[frame_idx] = None
        if eviction_scores:
            self.set_scores(eviction_scores)
        if evict:
            return self.evict_to_budget(protected_frames=protected_frames)
        return []

    def update(self, frame_indices, eviction_scores=None, protected_frames=None):
        evicted = []
        for frame_idx in frame_indices:
            evicted.extend(self.add(frame_idx, evict=False))
        if eviction_scores:
            self.set_scores(eviction_scores)
        evicted.extend(self.evict_to_budget(protected_frames=protected_frames))
        return evicted

    def set_scores(self, scores):
        for frame_idx, score in scores.items():
            frame_idx = int(frame_idx)
            if frame_idx in self._stats:
                self._stats[frame_idx]["score"] = float(score)

    def record_selection(self, frame_idx, overlap=1.0):
        frame_idx = int(frame_idx)
        if frame_idx not in self._stats:
            return
        overlap = max(float(overlap or 0.0), 0.0)
        stats = self._stats[frame_idx]
        stats["selected_count"] += 1
        stats["selection_overlap_sum"] += overlap
        stats["best_selection_overlap"] = max(stats["best_selection_overlap"], overlap)

    def evict_to_budget(self, protected_frames=None):
        if self.budget is None or self.policy == "unbounded":
            return []

        protected_frames = {
            int(frame_idx) for frame_idx in (protected_frames or [])
        } | self._pinned_frames
        evicted = []
        while len(self._frames) > self.budget:
            evictable = [
                frame_idx
                for frame_idx in self._frames.keys()
                if frame_idx not in protected_frames
            ]
            if not evictable:
                break

            if self.policy == "fifo":
                evicted_frame_idx = evictable[0]
            else:
                evicted_frame_idx = min(
                    evictable,
                    key=lambda idx: (
                        self._stats[idx].get("score", 0.0),
                        self._stats[idx]["insert_order"],
                    ),
                )

            self._frames.pop(evicted_frame_idx, None)
            self._stats.pop(evicted_frame_idx, None)
            evicted.append(evicted_frame_idx)
        return evicted

    def candidates(self, exclude_frames=None):
        exclude_frames = {int(frame_idx) for frame_idx in (exclude_frames or [])}
        return [
            frame_idx
            for frame_idx in self._frames.keys()
            if frame_idx not in exclude_frames
        ]

    def selected_count(self, frame_idx):
        return self._stats.get(int(frame_idx), {}).get("selected_count", 0)

    def __len__(self):
        return len(self._frames)


def rotation_distance(rotation_a, rotation_b):
    relative = rotation_a.T @ rotation_b
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.acos(cosine) / math.pi


def pose_distances(c2ws, frame_indices, target_indices, rotation_weight=2.0):
    frame_indices = list(frame_indices)
    target_indices = list(target_indices)
    if not frame_indices or not target_indices:
        return np.zeros((len(frame_indices), len(target_indices)), dtype=np.float64)

    c2ws = np.asarray(c2ws)
    frame_positions = c2ws[frame_indices, :3, 3]
    target_positions = c2ws[target_indices, :3, 3]
    position_dists = np.linalg.norm(
        frame_positions[:, None, :] - target_positions[None, :, :],
        axis=-1,
    )
    nonzero = position_dists[position_dists > 1e-8]
    position_scale = float(np.median(nonzero)) if nonzero.size else 1.0
    position_scale = max(position_scale, 1e-6)
    position_dists = position_dists / position_scale

    rotation_dists = np.zeros_like(position_dists)
    for row, frame_idx in enumerate(frame_indices):
        rotation_a = c2ws[frame_idx, :3, :3]
        for col, target_idx in enumerate(target_indices):
            rotation_b = c2ws[target_idx, :3, :3]
            rotation_dists[row, col] = rotation_distance(rotation_a, rotation_b)

    return position_dists + rotation_weight * rotation_dists


def cosine_distances(features):
    features = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    return 1.0 - similarities


def connected_components_from_threshold(pairwise_distances, threshold):
    num_items = pairwise_distances.shape[0]
    visited = np.zeros(num_items, dtype=bool)
    cluster_ids = np.full(num_items, -1, dtype=np.int64)
    clusters = []

    for start in range(num_items):
        if visited[start]:
            continue

        cluster_id = len(clusters)
        stack = [start]
        visited[start] = True
        members = []

        while stack:
            item = stack.pop()
            members.append(item)
            neighbors = np.flatnonzero(pairwise_distances[item] <= threshold)
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(int(neighbor))

        for member in members:
            cluster_ids[member] = cluster_id
        clusters.append(members)

    return cluster_ids, clusters


def estimate_cluster_threshold(pairwise_distances):
    finite = pairwise_distances[np.isfinite(pairwise_distances)]
    if finite.size == 0:
        return 0.0
    nearest = np.partition(pairwise_distances, 0, axis=1)[:, 0]
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size:
        return float(np.median(nearest))
    return float(np.median(finite))


def _feature_matrix(memory_frame_indices, features):
    missing = [idx for idx in memory_frame_indices if idx not in features]
    if missing:
        raise ValueError(f"Missing memory features for frames: {missing[:10]}")
    return np.stack([features[idx] for idx in memory_frame_indices])


def compute_rarity_irreplaceability_scores(
    memory_frame_indices,
    latent_features,
    pinned_frames=None,
    return_details=False,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(int(frame_idx) for frame_idx in (pinned_frames or []))
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    feature_matrix = _feature_matrix(memory_frame_indices, latent_features)
    pairwise = cosine_distances(feature_matrix)
    np.fill_diagonal(pairwise, np.inf)

    if len(memory_frame_indices) == 1:
        cluster_ids = np.zeros(1, dtype=np.int64)
        cluster_sizes = np.ones(1, dtype=np.float64)
        threshold = 0.0
        nearest_distances = np.ones(1, dtype=np.float64)
        nearest_indices = np.full(1, -1, dtype=np.int64)
    else:
        threshold = estimate_cluster_threshold(pairwise)
        cluster_pairwise = pairwise.copy()
        np.fill_diagonal(cluster_pairwise, 0.0)
        cluster_ids, clusters = connected_components_from_threshold(
            cluster_pairwise,
            threshold=threshold,
        )
        cluster_sizes = np.array([len(clusters[cluster_id]) for cluster_id in cluster_ids])
        nearest_indices = np.argmin(pairwise, axis=1)
        nearest_distances = pairwise[np.arange(len(memory_frame_indices)), nearest_indices]

    memory_count = float(len(memory_frame_indices))
    rarity = np.log((memory_count + 1.0) / np.maximum(cluster_sizes, 1.0))
    irreplaceability = nearest_distances

    scores = {}
    details = {}
    for index, frame_idx in enumerate(memory_frame_indices):
        score = float(rarity[index] * irreplaceability[index])
        if frame_idx in pinned_frames:
            score = float("inf")
        scores[frame_idx] = score
        details[frame_idx] = {
            "score": score,
            "rarity": float(rarity[index]),
            "irreplaceability": float(irreplaceability[index]),
            "cluster_id": int(cluster_ids[index]),
            "cluster_size": int(cluster_sizes[index]),
            "cluster_threshold": float(threshold),
            "nearest_frame": (
                None
                if nearest_indices[index] < 0
                else int(memory_frame_indices[int(nearest_indices[index])])
            ),
            "nearest_distance": float(nearest_distances[index]),
        }
    return (scores, details) if return_details else scores


def _feature_cosine_similarity(memory_frame_indices, features):
    feature_matrix = _feature_matrix(memory_frame_indices, features)
    norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    feature_matrix = feature_matrix / np.maximum(norms, 1e-12)
    return np.clip(feature_matrix @ feature_matrix.T, -1.0, 1.0)


def compute_slam_covisibility_scores(
    memory_frame_indices,
    c2ws,
    pinned_frames=None,
    latent_features=None,
    n_other_observers=3,
    covisibility_threshold=0.65,
    visual_weight=0.35,
    geometry_weight=0.65,
    return_details=False,
):
    memory_frame_indices = list(memory_frame_indices)
    pinned_frames = set(int(frame_idx) for frame_idx in (pinned_frames or []))
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    pose_distance = pose_distances(c2ws, memory_frame_indices, memory_frame_indices)
    geom_similarity = np.exp(-pose_distance)
    np.fill_diagonal(geom_similarity, 0.0)

    components = [(geometry_weight, geom_similarity)]
    if latent_features is not None:
        visual_similarity = _feature_cosine_similarity(memory_frame_indices, latent_features)
        visual_similarity = np.maximum(visual_similarity, 0.0)
        np.fill_diagonal(visual_similarity, 0.0)
        components.append((visual_weight, visual_similarity))

    total_weight = sum(weight for weight, _ in components)
    covisibility = sum(weight * matrix for weight, matrix in components) / max(total_weight, 1e-12)
    np.fill_diagonal(covisibility, 0.0)

    scores = {}
    details = {}
    for row, frame_idx in enumerate(memory_frame_indices):
        row_values = covisibility[row]
        observer_indices = np.flatnonzero(row_values >= covisibility_threshold)
        covisible_observers = int(observer_indices.size)
        redundancy_ratio = min(covisible_observers / max(float(n_other_observers), 1.0), 1.0)

        if row_values.size:
            nearest_index = int(np.argmax(row_values))
            nearest_frame = int(memory_frame_indices[nearest_index])
            max_covisibility = float(row_values[nearest_index])
        else:
            nearest_frame = None
            max_covisibility = 0.0

        marginal_contribution = 1.0 / (covisible_observers + 1.0)
        unique_bonus = 1.0 - max_covisibility
        score = (1.0 - redundancy_ratio) + 0.5 * marginal_contribution + 0.25 * unique_bonus
        if frame_idx in pinned_frames:
            score = float("inf")

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "redundancy_ratio": float(redundancy_ratio),
            "covisible_observers": covisible_observers,
            "max_covisibility": float(max_covisibility),
            "nearest_covisible_frame": nearest_frame,
            "marginal_contribution": float(marginal_contribution),
            "unique_bonus": float(unique_bonus),
            "covisibility_threshold": float(covisibility_threshold),
            "n_other_observers": int(n_other_observers),
        }

    return (scores, details) if return_details else scores


class DinoFeatureExtractor:
    """Calibrated visual-similarity features for K_vis (Sec. 3.1).

    Mirrors MemCam's ``VisualMemoryFeatureExtractor`` DINO half so both
    backbones share the same K_vis definition for the cross-backbone
    comparison. Lazily imports torch/transformers so this module stays
    importable (and unit-testable) without those dependencies present.
    """

    def __init__(self, dino_model_name="facebook/dinov2-base", device="cuda", batch_size=16):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.processor = AutoImageProcessor.from_pretrained(dino_model_name)
        self.model = AutoModel.from_pretrained(dino_model_name).eval().to(self.device)

    def encode_pil_images(self, images):
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                batch_features = getattr(outputs, "pooler_output", None)
                if batch_features is None:
                    batch_features = outputs.last_hidden_state[:, 0]
                batch_features = self.torch.nn.functional.normalize(batch_features.float(), dim=-1)
                features.append(batch_features.detach().cpu().numpy())

        if features:
            return np.concatenate(features, axis=0)
        return np.zeros((0, 0), dtype=np.float32)


def _feature_cosine_similarity_cross(left_frame_indices, right_frame_indices, features):
    left_matrix = _feature_matrix(left_frame_indices, features)
    right_matrix = _feature_matrix(right_frame_indices, features)
    left_norms = np.linalg.norm(left_matrix, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right_matrix, axis=1, keepdims=True)
    left_matrix = left_matrix / np.maximum(left_norms, 1e-12)
    right_matrix = right_matrix / np.maximum(right_norms, 1e-12)
    return np.clip(left_matrix @ right_matrix.T, -1.0, 1.0)


def historical_query_medoids(memory_frame_indices, dino_features, rarity_neighbors=3):
    """Q_hist (Sec. 3.1): one DINO-cluster medoid per distinct scene mode.

    Backbone-agnostic -- reuses the same connected-components clustering as
    ``compute_rarity_irreplaceability_scores`` so "distinct scene mode" means
    the same thing everywhere in this module. Each cluster contributes
    exactly one query regardless of its size: a corridor revisited 100 times
    is one query, same as a room visited once -- controlled rather than
    frequency-proportional weighting.
    """
    memory_frame_indices = list(memory_frame_indices)
    if len(memory_frame_indices) == 1:
        return [memory_frame_indices[0]], [[0]]

    dino_matrix = _feature_matrix(memory_frame_indices, dino_features)
    dino_pairwise = cosine_distances(dino_matrix)
    np.fill_diagonal(dino_pairwise, np.inf)
    threshold = estimate_cluster_threshold(dino_pairwise)

    cluster_pairwise = dino_pairwise.copy()
    np.fill_diagonal(cluster_pairwise, 0.0)
    _, clusters = connected_components_from_threshold(cluster_pairwise, threshold=threshold)

    medoid_positions = []
    for members in clusters:
        if len(members) == 1:
            medoid_positions.append(members[0])
            continue
        sub_distances = cluster_pairwise[np.ix_(members, members)]
        total_distance = sub_distances.sum(axis=1)
        medoid_positions.append(members[int(np.argmin(total_distance))])

    medoid_frame_indices = [memory_frame_indices[position] for position in medoid_positions]
    return medoid_frame_indices, clusters


def compute_marginal_coverage_eviction_scores(
    memory_frame_indices,
    budget,
    hist_query_frame_indices,
    hist_geo_matrix,
    dino_features,
    ctrl_query_frame_indices=None,
    ctrl_geo_matrix=None,
    forced_keep_frames=None,
    alpha=0.65,
    lambda_hist=None,
    gamma=0.25,
    return_details=False,
):
    """Marginal Coverage Eviction (MCE): the paper-faithful set-coverage policy.

    Backbone-agnostic core: takes K_geo as precomputed (query x candidate)
    matrices rather than computing them itself, since VMem's geometric
    support comes from rendering its surfel index (see
    ``VMemPipeline._surfel_geometric_support``), not from a generic pose
    kernel. K_vis is computed here from DINO features, which is backbone-
    independent.

    - Query set Q = Q_hist ∪ Q_ctrl (Sec. 3.1). ``hist_query_frame_indices``
      are the DINO-cluster medoids (see ``historical_query_medoids``),
      weighted uniformly (``lambda_hist / J``). Q_ctrl is the known future
      camera path, weighted with exponential horizon decay
      ``w_h ∝ exp(-gamma * h)``; omitted when no future controls are known
      (``lambda_hist`` then defaults to 1).
    - Kernel K(q, m) = alpha * K_geo(q, m) + (1 - alpha) * K_vis(q, m), an
      explicit convex combination (Eq. 6), not a product. Future queries have
      no realized appearance yet, so they use K_geo alone.
    - Eviction is reverse deletion (Algorithm 1): repeatedly remove
      argmin_i Delta_i(P) from the full candidate pool P, recomputing exact
      deletion marginals after each removal, until |P| <= B.
    """
    memory_frame_indices = list(memory_frame_indices)
    hist_query_frame_indices = list(hist_query_frame_indices)
    ctrl_query_frame_indices = list(ctrl_query_frame_indices or [])
    forced_keep_frames = set(forced_keep_frames or [])

    if budget is None:
        raise ValueError("mce requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("mce budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}
    if len(set(memory_frame_indices)) != len(memory_frame_indices):
        raise ValueError("mce candidates must be unique")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("mce alpha must be in [0, 1]")
    if lambda_hist is not None and not 0.0 <= lambda_hist <= 1.0:
        raise ValueError("mce lambda_hist must be in [0, 1]")
    if gamma < 0:
        raise ValueError("mce gamma must be non-negative")

    num_candidates = len(memory_frame_indices)
    candidate_set = set(memory_frame_indices)
    unknown_forced = forced_keep_frames - candidate_set
    if unknown_forced:
        raise ValueError(f"mce forced-keep frames are not candidates: {sorted(unknown_forced)[:10]}")
    if len(forced_keep_frames) > budget:
        raise ValueError("mce has more forced frames than its budget")

    num_hist = len(hist_query_frame_indices)
    hist_geo_matrix = np.asarray(hist_geo_matrix, dtype=np.float64).reshape(num_hist, num_candidates)
    num_ctrl = len(ctrl_query_frame_indices)
    if num_ctrl:
        ctrl_geo_matrix = np.asarray(ctrl_geo_matrix, dtype=np.float64).reshape(num_ctrl, num_candidates)

    selected_limit = min(int(budget), num_candidates)

    lambda_eff = (
        1.0 if not num_ctrl else (0.5 if lambda_hist is None else float(lambda_hist))
    )

    # --- Kernel (Eq. 6): explicit convex combination, not a product --------
    hist_vis_cosine = _feature_cosine_similarity_cross(
        hist_query_frame_indices, memory_frame_indices, dino_features
    )
    hist_vis = np.clip((hist_vis_cosine + 1.0) / 2.0, 0.0, 1.0)  # calibrate [-1,1] -> [0,1]
    hist_geo = np.clip(hist_geo_matrix, 0.0, 1.0)
    hist_kernel = np.clip(alpha * hist_geo + (1.0 - alpha) * hist_vis, 0.0, 1.0 - 1e-6)
    hist_weights = np.full(num_hist, lambda_eff / max(num_hist, 1), dtype=np.float64)

    if num_ctrl:
        # Future queries have no realized appearance yet: K_geo alone.
        ctrl_kernel = np.clip(ctrl_geo_matrix, 0.0, 1.0 - 1e-6)
        horizons = np.arange(1, num_ctrl + 1, dtype=np.float64)
        raw_ctrl_weights = np.exp(-float(gamma) * horizons)
        ctrl_weights = (1.0 - lambda_eff) * raw_ctrl_weights / np.sum(raw_ctrl_weights)
    else:
        ctrl_kernel = np.zeros((0, num_candidates), dtype=np.float64)
        ctrl_weights = np.zeros(0, dtype=np.float64)

    kernel = np.vstack([hist_kernel, ctrl_kernel])
    weights = np.concatenate([hist_weights, ctrl_weights])
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("mce query weights must have positive total mass")
    weights = weights / weight_sum

    # --- Algorithm 1: reverse deletion with exact recomputed marginals -----
    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = {frame_to_col[frame_idx] for frame_idx in forced_keep_frames}
    remaining_cols = list(range(num_candidates))
    one_minus_kernel = 1.0 - kernel
    pool_product = np.prod(one_minus_kernel, axis=1)  # P_q over the full initial pool

    removal_order = []
    removal_marginals = {}
    while len(remaining_cols) > selected_limit:
        remaining = np.array(remaining_cols)
        denom = np.maximum(one_minus_kernel[:, remaining], 1e-12)
        marginals = np.sum(
            (weights[:, None] * kernel[:, remaining] * pool_product[:, None]) / denom,
            axis=0,
        )
        eviction_candidates = [
            (float(marginals[position]), col)
            for position, col in enumerate(remaining_cols)
            if col not in forced_cols
        ]
        if not eviction_candidates:
            break
        loss, evict_col = min(eviction_candidates, key=lambda item: item[0])
        removal_order.append(evict_col)
        removal_marginals[evict_col] = loss
        pool_product = pool_product / np.maximum(one_minus_kernel[:, evict_col], 1e-12)
        remaining_cols.remove(evict_col)

    selected_cols = remaining_cols
    selected_frame_set = {memory_frame_indices[col] for col in selected_cols}

    # Final leave-one-out marginal for each survivor -- reported "value",
    # not what drove the eviction decisions.
    final_uncovered = (
        np.prod(one_minus_kernel[:, selected_cols], axis=1) if selected_cols else np.ones(kernel.shape[0])
    )
    survivor_marginals = {}
    for col in selected_cols:
        denom = np.maximum(one_minus_kernel[:, col], 1e-12)
        survivor_marginals[col] = float(np.sum(weights * kernel[:, col] * final_uncovered / denom))
    coverage_value = float(np.sum(weights * (1.0 - final_uncovered)))

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            # Positive scores for survivors, negative for evictions: the
            # generic FrameMemoryBuffer eviction loop (which removes the
            # global min-score item repeatedly from this one static score
            # dict) then drains the pool to exactly this function's already-
            # computed reverse-deletion result without needing to re-invoke
            # this function per removal step.
            score = 1.0 + survivor_marginals.get(col, 0.0)
        else:
            score = -1.0 - removal_marginals.get(col, 0.0)

        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "mce_selected": bool(selected),
            "mce_forced_keep": bool(forced),
            "mce_removal_rank": removal_order.index(col) if col in removal_order else None,
            "mce_removal_marginal": removal_marginals.get(col),
            "mce_survivor_marginal": survivor_marginals.get(col),
            "mce_coverage_value": coverage_value,
            "mce_alpha": float(alpha),
            "mce_lambda": float(lambda_eff),
            "mce_gamma": float(gamma),
            "mce_num_hist_queries": num_hist,
            "mce_num_ctrl_queries": num_ctrl,
            "mce_hist_query_frames": [int(f) for f in hist_query_frame_indices],
        }

    return (scores, details) if return_details else scores


def compute_kcenter_coreset_scores(
    memory_frame_indices,
    c2ws,
    budget,
    archive_frame_indices=None,
    forced_keep_frames=None,
    dino_features=None,
    visual_weight=0.5,
    pose_weight=0.5,
    return_details=False,
):
    """Greedy k-center coreset selection: representation-agnostic, ported
    directly from MemCam's ``compute_kcenter_coreset_scores`` since it only
    needs a distance/similarity between candidates, not a backbone-specific
    cue. Repeatedly adds the candidate nearest to the archive point currently
    farthest from any already-selected center, minimizing the maximum
    distance from any archive point to its nearest retained memory item.
    """
    memory_frame_indices = list(memory_frame_indices)
    archive_frame_indices = list(archive_frame_indices or memory_frame_indices)
    forced_keep_frames = set(forced_keep_frames or [])
    if budget is None:
        raise ValueError("kcenter_coreset requires an explicit memory budget")
    if budget <= 0:
        raise ValueError("kcenter_coreset budget must be positive")
    if not memory_frame_indices:
        return ({}, {}) if return_details else {}

    use_visual = dino_features is not None and float(visual_weight) > 0.0
    if use_visual:
        missing_features = [
            frame_idx
            for frame_idx in set(memory_frame_indices) | set(archive_frame_indices)
            if frame_idx not in dino_features
        ]
        if missing_features:
            raise ValueError(f"Missing k-center DINO features for frames: {missing_features[:10]}")

    if len(memory_frame_indices) <= budget:
        scores = {
            frame_idx: float("inf") if frame_idx in forced_keep_frames else 1.0
            for frame_idx in memory_frame_indices
        }
        details = {
            frame_idx: {
                "score": scores[frame_idx],
                "kcenter_selected": True,
                "kcenter_forced_keep": frame_idx in forced_keep_frames,
                "kcenter_rank": index,
                "kcenter_radius": 0.0,
                "kcenter_archive_size": len(archive_frame_indices),
            }
            for index, frame_idx in enumerate(memory_frame_indices)
        }
        return (scores, details) if return_details else scores

    components = []
    if use_visual:
        visual_similarity = _feature_cosine_similarity_cross(
            archive_frame_indices, memory_frame_indices, dino_features
        )
        visual_distance = np.clip((1.0 - visual_similarity) / 2.0, 0.0, 1.0)
        components.append((float(visual_weight), visual_distance))

    if pose_weight:
        pose_distance = pose_distances(c2ws, archive_frame_indices, memory_frame_indices)
        pose_distance = 1.0 - np.exp(-pose_distance)
        components.append((float(pose_weight), pose_distance))

    if not components:
        raise ValueError("kcenter_coreset needs at least one positive distance component")

    total_weight = max(sum(weight for weight, _ in components), 1e-12)
    distance = sum(weight * matrix for weight, matrix in components) / total_weight

    frame_to_col = {frame_idx: col for col, frame_idx in enumerate(memory_frame_indices)}
    forced_cols = [
        frame_to_col[frame_idx] for frame_idx in memory_frame_indices if frame_idx in forced_keep_frames
    ]

    selected_cols = []
    selected_set = set()
    for col in forced_cols:
        if col not in selected_set:
            selected_set.add(col)
            selected_cols.append(col)

    if selected_cols:
        covered_distance = np.min(distance[:, selected_cols], axis=1)
    else:
        first_col = int(np.argmin(np.mean(distance, axis=0)))
        selected_set.add(first_col)
        selected_cols.append(first_col)
        covered_distance = distance[:, first_col].copy()

    while len(selected_cols) < min(int(budget), len(memory_frame_indices)):
        farthest_archive_row = int(np.argmax(covered_distance))
        candidate_order = np.argsort(distance[farthest_archive_row])
        best_col = None
        for col in candidate_order:
            col = int(col)
            if col not in selected_set:
                best_col = col
                break
        if best_col is None:
            break
        selected_set.add(best_col)
        selected_cols.append(best_col)
        covered_distance = np.minimum(covered_distance, distance[:, best_col])

    selected_frames = [memory_frame_indices[col] for col in selected_cols]
    selected_frame_set = set(selected_frames)
    current_radius = float(np.max(covered_distance)) if covered_distance.size else 0.0

    removal_radius_increases = {}
    for col in selected_cols:
        other_cols = [other for other in selected_cols if other != col]
        if other_cols:
            without_col = np.min(distance[:, other_cols], axis=1)
            without_radius = float(np.max(without_col))
        else:
            without_radius = float("inf")
        removal_radius_increases[col] = without_radius - current_radius

    scores = {}
    details = {}
    for col, frame_idx in enumerate(memory_frame_indices):
        selected = frame_idx in selected_frame_set
        forced = frame_idx in forced_keep_frames
        if forced:
            score = float("inf")
        elif selected:
            score = 1.0 + max(float(removal_radius_increases.get(col, 0.0)), 0.0)
        else:
            score = -1.0

        rank = selected_frames.index(frame_idx) if selected else None
        scores[frame_idx] = float(score)
        details[frame_idx] = {
            "score": float(score),
            "kcenter_selected": bool(selected),
            "kcenter_forced_keep": bool(forced),
            "kcenter_rank": rank,
            "kcenter_radius": current_radius,
            "kcenter_removal_radius_increase": (
                float(removal_radius_increases.get(col, 0.0)) if selected else 0.0
            ),
            "kcenter_archive_size": len(archive_frame_indices),
            "kcenter_visual_weight": float(visual_weight if use_visual else 0.0),
            "kcenter_pose_weight": float(pose_weight),
        }

    return (scores, details) if return_details else scores
