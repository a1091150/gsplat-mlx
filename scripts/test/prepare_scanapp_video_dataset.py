#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ScanApp RGB-video + JSONL captures into the per-frame dataset layout used by the depth trainers."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--image-extension", default="jpg", choices=("jpg", "png"))
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--copy-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl_records(metadata_dir: Path) -> list[tuple[dict, Path, int]]:
    records: list[tuple[dict, Path, int]] = []
    for path in sorted(metadata_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                records.append((json.loads(stripped), path, line_no))
    if not records:
        raise RuntimeError(f"No JSONL records found in {metadata_dir}")
    return records


def resolve_relative_path(data_dir: Path, rel_or_abs: str) -> Path:
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return data_dir / path


def find_rgb_video(data_dir: Path, records: list[tuple[dict, Path, int]]) -> Path:
    first = records[0][0]
    candidates: list[str] = []
    rgb = first.get("rgb")
    if isinstance(rgb, dict) and isinstance(rgb.get("path"), str):
        candidates.append(rgb["path"])
    if isinstance(first.get("image"), str):
        candidates.append(first["image"])

    session_path = data_dir / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
        layout = session.get("dataset_layout")
        if isinstance(layout, dict) and isinstance(layout.get("rgb"), str):
            candidates.append(layout["rgb"])

    for candidate in candidates:
        path = resolve_relative_path(data_dir, candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find RGB video from metadata/session candidates: {candidates}")


def run_ffmpeg_extract(args: argparse.Namespace, rgb_video: Path, images_dir: Path) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    ext = args.image_extension
    pattern = images_dir / f"frame_%06d.{ext}"
    command = [
        args.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(rgb_video),
        "-vsync",
        "0",
        "-start_number",
        "1",
    ]
    if ext == "jpg":
        command += ["-q:v", str(args.jpeg_quality)]
    command.append(str(pattern))
    subprocess.run(command, check=True)


def select_records(
    records: list[tuple[dict, Path, int]],
    start_index: int,
    frame_step: int,
    max_frames: int,
) -> list[tuple[dict, Path, int]]:
    if frame_step <= 0:
        raise ValueError("--frame-step must be positive")
    if start_index < 0:
        raise ValueError("--start-index must be non-negative")
    selected = records[start_index::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    return selected


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_file:
        shutil.copy2(src, dst)
        return "copy"
    try:
        dst.symlink_to(src)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_metadata_and_depth_assets(
    args: argparse.Namespace,
    records: list[tuple[dict, Path, int]],
    out_dir: Path,
    total_video_frames: int,
) -> dict:
    metadata_dir = out_dir / "metadata"
    depth_dir = out_dir / "depth"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    linked_depth = 0
    copied_depth = 0
    linked_confidence = 0
    copied_confidence = 0

    selected = select_records(records, args.start_index, args.frame_step, args.max_frames)
    for raw, source_jsonl, source_line in selected:
        frame_index = int(raw.get("frame_index", written + 1))
        frame_name = str(raw.get("frame_name", f"frame_{frame_index:06d}"))
        image_name = f"frame_{frame_index:06d}.{args.image_extension}"
        image_path = out_dir / "images" / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Expected extracted RGB frame not found: {image_path}")

        converted = dict(raw)
        converted["image"] = f"images/{image_name}"
        converted["source_jsonl"] = str(source_jsonl)
        converted["source_jsonl_line"] = source_line
        converted["source_capture_output"] = raw.get("capture_output", "rgb_video")
        converted["capture_output"] = "per_frame_image"

        depth = converted.get("depth")
        if not isinstance(depth, dict) or not isinstance(depth.get("path"), str):
            raise RuntimeError(f"Missing per-frame depth path for {frame_name} in {source_jsonl}:{source_line}")
        depth_src = resolve_relative_path(args.data, depth["path"])
        if not depth_src.exists():
            raise FileNotFoundError(f"Depth file from metadata does not exist: {depth_src}")
        depth_dst = depth_dir / depth_src.name
        mode = link_or_copy(depth_src, depth_dst, args.copy_depth)
        linked_depth += mode == "symlink"
        copied_depth += mode == "copy"
        depth = dict(depth)
        depth["path"] = f"depth/{depth_dst.name}"

        confidence_rel = depth.get("confidence_path")
        if isinstance(confidence_rel, str):
            confidence_src = resolve_relative_path(args.data, confidence_rel)
            if confidence_src.exists():
                confidence_dst = depth_dir / confidence_src.name
                mode = link_or_copy(confidence_src, confidence_dst, args.copy_depth)
                linked_confidence += mode == "symlink"
                copied_confidence += mode == "copy"
                depth["confidence_path"] = f"depth/{confidence_dst.name}"

        converted["depth"] = depth
        (metadata_dir / f"{frame_name}.json").write_text(
            json.dumps(converted, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        written += 1

    session_src = args.data / "session.json"
    if session_src.exists():
        shutil.copy2(session_src, out_dir / "source_session.json")

    return {
        "jsonl_records": len(records),
        "video_frames": total_video_frames,
        "selected_frames": len(selected),
        "written_metadata": written,
        "depth_symlinks": linked_depth,
        "depth_copies": copied_depth,
        "confidence_symlinks": linked_confidence,
        "confidence_copies": copied_confidence,
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.out_dir} (pass --overwrite to replace it)")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    metadata_dir = args.data / "metadata"
    records = load_jsonl_records(metadata_dir)
    rgb_video = find_rgb_video(args.data, records)
    images_dir = args.out_dir / "images"
    run_ffmpeg_extract(args, rgb_video, images_dir)

    extracted_images = sorted(images_dir.glob(f"*.{args.image_extension}"))
    if len(extracted_images) != len(records):
        raise RuntimeError(
            f"Extracted RGB frame count mismatch: images={len(extracted_images)} jsonl_records={len(records)}"
        )

    summary = write_metadata_and_depth_assets(args, records, args.out_dir, len(extracted_images))
    summary.update(
        {
            "source_data": str(args.data),
            "rgb_video": str(rgb_video),
            "out_dir": str(args.out_dir),
            "image_extension": args.image_extension,
            "copy_depth": args.copy_depth,
        }
    )
    (args.out_dir / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "prepared ScanApp video dataset "
        f"frames={summary['written_metadata']} rgb_frames={summary['video_frames']} out={args.out_dir}"
    )


if __name__ == "__main__":
    main()
