"""Read-only adapter for the Context-as-Memory dataset.

Expected dataset layout:

    root/
      captions.txt
      frames/<scene_id>/<frame_idx>.png
      jsons/<scene_id>.json
      overlap_labels/<scene_id>/<frame_idx>.json

This module deliberately does not import torch or VMem. It is the thin indexing
layer we can trust before connecting dataset trajectories to generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


_CAPTION_PATH_RE = re.compile(
    r"^(?P<scene_id>[^/]+)/(?P<start>\d+)_(?P<end>\d+)\.mp4$"
)


@dataclass(frozen=True)
class CameraFrame:
    index: int
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float]
    scale: Tuple[float, float, float]


@dataclass(frozen=True)
class OverlapLabel:
    frame_index: int
    overlapping_frames: Tuple[int, ...]

    def past_only(self) -> "OverlapLabel":
        return OverlapLabel(
            frame_index=self.frame_index,
            overlapping_frames=tuple(
                idx for idx in self.overlapping_frames if idx < self.frame_index
            ),
        )


@dataclass(frozen=True)
class CaptionSegment:
    scene_id: str
    start_frame: int
    end_frame: int
    video_path: str
    caption: str

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True)
class SequenceValidationSummary:
    scene_id: str
    num_frames: int
    num_camera_entries: int
    num_overlap_files: int
    num_caption_segments: int
    missing_camera_indices: Tuple[int, ...]
    missing_overlap_indices: Tuple[int, ...]
    overlap_references_missing_frames: Tuple[Tuple[int, int], ...]

    @property
    def ok(self) -> bool:
        return (
            not self.missing_camera_indices
            and not self.missing_overlap_indices
            and not self.overlap_references_missing_frames
        )


def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_vector3(value: object, *, field_name: str, path: Path, index: int) -> Tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(
            f"{path}: camera frame {index} field {field_name!r} must be a length-3 list"
        )
    return (float(value[0]), float(value[1]), float(value[2]))


def _parse_caption_line(line: str, *, line_number: int, path: Path) -> CaptionSegment:
    stripped = line.strip()
    if not stripped:
        raise ValueError(f"{path}: caption line {line_number} is empty")

    try:
        video_path, caption = stripped.split(maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"{path}: caption line {line_number} must start with '<scene>/<start>_<end>.mp4'"
        ) from exc

    match = _CAPTION_PATH_RE.match(video_path)
    if match is None:
        raise ValueError(
            f"{path}: caption line {line_number} has unexpected video path {video_path!r}"
        )

    return CaptionSegment(
        scene_id=match.group("scene_id"),
        start_frame=int(match.group("start")),
        end_frame=int(match.group("end")),
        video_path=video_path,
        caption=caption,
    )


def _numeric_png_indices(frames_dir: Path) -> Tuple[int, ...]:
    indices: List[int] = []
    for path in frames_dir.iterdir():
        if path.suffix.lower() != ".png":
            continue
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return tuple(sorted(indices))


def _parse_camera_json(path: Path) -> Dict[int, CameraFrame]:
    raw = _read_json(path)
    if not isinstance(raw, Mapping) or "CineCameraActor" not in raw:
        raise ValueError(f"{path}: expected top-level key 'CineCameraActor'")

    camera_actor = raw["CineCameraActor"]
    if not isinstance(camera_actor, Mapping):
        raise ValueError(f"{path}: 'CineCameraActor' must be an object")

    frames: Dict[int, CameraFrame] = {}
    for raw_index, raw_entry in camera_actor.items():
        index = int(raw_index)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"{path}: camera frame {index} must be an object")
        frames[index] = CameraFrame(
            index=index,
            position=_parse_vector3(
                raw_entry.get("position"),
                field_name="position",
                path=path,
                index=index,
            ),
            rotation=_parse_vector3(
                raw_entry.get("rotation"),
                field_name="rotation",
                path=path,
                index=index,
            ),
            scale=_parse_vector3(
                raw_entry.get("scale"),
                field_name="scale",
                path=path,
                index=index,
            ),
        )
    return frames


def _parse_overlap_json(path: Path) -> OverlapLabel:
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected overlap label object")

    frame_index = int(raw.get("frame_index"))
    raw_overlaps = raw.get("overlapping_frames")
    if not isinstance(raw_overlaps, Sequence) or isinstance(raw_overlaps, (str, bytes)):
        raise ValueError(f"{path}: expected list field 'overlapping_frames'")

    return OverlapLabel(
        frame_index=frame_index,
        overlapping_frames=tuple(int(idx) for idx in raw_overlaps),
    )


class ContextMemorySequence:
    def __init__(
        self,
        *,
        scene_id: str,
        frames_dir: Path,
        camera_json_path: Path,
        overlap_dir: Optional[Path],
        caption_segments: Sequence[CaptionSegment],
    ) -> None:
        self.scene_id = scene_id
        self.frames_dir = frames_dir
        self.camera_json_path = camera_json_path
        self.overlap_dir = overlap_dir
        self.caption_segments = tuple(caption_segments)

        self._frame_indices: Optional[Tuple[int, ...]] = None
        self._camera_frames: Optional[Dict[int, CameraFrame]] = None
        self._overlap_cache: Dict[int, OverlapLabel] = {}

    @property
    def frame_indices(self) -> Tuple[int, ...]:
        if self._frame_indices is None:
            self._frame_indices = _numeric_png_indices(self.frames_dir)
        return self._frame_indices

    @property
    def camera_frames(self) -> Mapping[int, CameraFrame]:
        if self._camera_frames is None:
            self._camera_frames = _parse_camera_json(self.camera_json_path)
        return self._camera_frames

    def frame_path(self, frame_index: int) -> Path:
        return self.frames_dir / f"{frame_index:04d}.png"

    def camera(self, frame_index: int) -> CameraFrame:
        try:
            return self.camera_frames[frame_index]
        except KeyError as exc:
            raise KeyError(f"{self.scene_id}: no camera metadata for frame {frame_index}") from exc

    def caption_segments_for_frame(self, frame_index: int) -> Tuple[CaptionSegment, ...]:
        return tuple(
            segment for segment in self.caption_segments if segment.contains(frame_index)
        )

    def overlap_label(self, frame_index: int, *, past_only: bool = False) -> OverlapLabel:
        if self.overlap_dir is None:
            raise FileNotFoundError(f"{self.scene_id}: no overlap label directory")

        if frame_index not in self._overlap_cache:
            path = self.overlap_dir / f"{frame_index}.json"
            if not path.exists():
                raise FileNotFoundError(f"{self.scene_id}: missing overlap label {path}")
            label = _parse_overlap_json(path)
            if label.frame_index != frame_index:
                raise ValueError(
                    f"{path}: frame_index={label.frame_index} does not match filename {frame_index}"
                )
            self._overlap_cache[frame_index] = label

        label = self._overlap_cache[frame_index]
        return label.past_only() if past_only else label

    def iter_frame_records(self, *, include_overlaps: bool = False) -> Iterator[dict]:
        for frame_index in self.frame_indices:
            record = {
                "scene_id": self.scene_id,
                "frame_index": frame_index,
                "frame_path": self.frame_path(frame_index),
                "camera": self.camera(frame_index),
                "caption_segments": self.caption_segments_for_frame(frame_index),
            }
            if include_overlaps:
                record["overlap_label"] = self.overlap_label(frame_index)
            yield record

    def validate(self, *, check_overlap_references: bool = True) -> SequenceValidationSummary:
        frame_index_set = set(self.frame_indices)
        camera_index_set = set(self.camera_frames.keys())

        overlap_indices: Tuple[int, ...] = ()
        overlap_refs_missing: List[Tuple[int, int]] = []
        if self.overlap_dir is not None and self.overlap_dir.exists():
            overlap_indices = tuple(
                sorted(int(path.stem) for path in self.overlap_dir.glob("*.json"))
            )
            if check_overlap_references:
                for frame_index in overlap_indices:
                    label = self.overlap_label(frame_index)
                    for overlap_index in label.overlapping_frames:
                        if overlap_index not in frame_index_set:
                            overlap_refs_missing.append((frame_index, overlap_index))

        return SequenceValidationSummary(
            scene_id=self.scene_id,
            num_frames=len(self.frame_indices),
            num_camera_entries=len(camera_index_set),
            num_overlap_files=len(overlap_indices),
            num_caption_segments=len(self.caption_segments),
            missing_camera_indices=tuple(
                sorted(frame_index_set.difference(camera_index_set))
            ),
            missing_overlap_indices=tuple(
                sorted(frame_index_set.difference(set(overlap_indices)))
            ),
            overlap_references_missing_frames=tuple(overlap_refs_missing),
        )


class ContextMemoryDataset:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.frames_root = self.root / "frames"
        self.jsons_root = self.root / "jsons"
        self.overlap_root = self.root / "overlap_labels"
        self.captions_path = self.root / "captions.txt"

    def _require_layout(self) -> None:
        missing = [
            path
            for path in (self.frames_root, self.jsons_root, self.overlap_root)
            if not path.exists()
        ]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"Context-as-Memory dataset layout missing: {joined}")

    def caption_segments(self) -> Tuple[CaptionSegment, ...]:
        if not self.captions_path.exists():
            return ()

        segments: List[CaptionSegment] = []
        with self.captions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                segments.append(
                    _parse_caption_line(
                        line,
                        line_number=line_number,
                        path=self.captions_path,
                    )
                )
        return tuple(segments)

    def scene_ids(self) -> Tuple[str, ...]:
        self._require_layout()
        frame_scene_ids = {
            path.name for path in self.frames_root.iterdir() if path.is_dir()
        }
        json_scene_ids = {path.stem for path in self.jsons_root.glob("*.json")}
        return tuple(sorted(frame_scene_ids.intersection(json_scene_ids)))

    def sequence(self, scene_id: str) -> ContextMemorySequence:
        self._require_layout()
        frames_dir = self.frames_root / scene_id
        camera_json_path = self.jsons_root / f"{scene_id}.json"
        overlap_dir = self.overlap_root / scene_id

        if not frames_dir.exists():
            raise FileNotFoundError(f"{scene_id}: missing frames directory {frames_dir}")
        if not camera_json_path.exists():
            raise FileNotFoundError(
                f"{scene_id}: missing camera json {camera_json_path}"
            )

        captions = [
            segment for segment in self.caption_segments() if segment.scene_id == scene_id
        ]
        return ContextMemorySequence(
            scene_id=scene_id,
            frames_dir=frames_dir,
            camera_json_path=camera_json_path,
            overlap_dir=overlap_dir if overlap_dir.exists() else None,
            caption_segments=captions,
        )

    def sequences(self, scene_ids: Optional[Iterable[str]] = None) -> Iterator[ContextMemorySequence]:
        selected_scene_ids = tuple(scene_ids) if scene_ids is not None else self.scene_ids()
        for scene_id in selected_scene_ids:
            yield self.sequence(scene_id)
