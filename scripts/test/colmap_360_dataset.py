#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ColmapTrainingCamera:
    index: int
    viewmat: np.ndarray
    K: np.ndarray
    image_path: Path
    raw_width: int
    raw_height: int
    camera_id: int
    image_name: str


@dataclass(frozen=True)
class ColmapScene:
    cameras: list[ColmapTrainingCamera]
    points: np.ndarray
    colors: np.ndarray
    raw_point_count: int
    scene_scale: float
    transform: np.ndarray
    data_factor: int
    test_every: int
    normalize_world_space: bool


def _get_rel_paths(root: Path) -> list[str]:
    return [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not path.name.startswith(".")
    ]


def _colmap_dir(data_dir: Path) -> Path:
    sparse0 = data_dir / "sparse" / "0"
    if sparse0.exists():
        return sparse0
    sparse = data_dir / "sparse"
    if sparse.exists():
        return sparse
    raise FileNotFoundError(f"COLMAP sparse directory not found in {data_dir}")


def similarity_from_cameras(c2w: np.ndarray) -> np.ndarray:
    t = c2w[:, :3, 3]
    r = c2w[:, :3, :3]
    ups = np.sum(r * np.array([0.0, -1.0, 0.0], dtype=np.float32), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    c = float((up_camspace * world_up).sum())
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float32,
    )
    if c > -1:
        r_align = np.eye(3, dtype=np.float32) + skew + (skew @ skew) / (1 + c)
    else:
        r_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    r = r_align @ r
    fwds = np.sum(r * np.array([0.0, 0.0, 1.0], dtype=np.float32), axis=-1)
    t = (r_align @ t[..., None])[..., 0]
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
    translate = -np.median(nearest, axis=0)
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = translate
    transform[:3, :3] = r_align
    transform[:3, :] *= scale
    return transform


def align_principal_axes(points: np.ndarray) -> np.ndarray:
    centroid = np.median(points, axis=0)
    centered = points - centroid
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = eigenvectors.T.astype(np.float32)
    transform[:3, 3] = (-transform[:3, :3] @ centroid).astype(np.float32)
    return transform


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def transform_cameras(matrix: np.ndarray, camtoworlds: np.ndarray) -> np.ndarray:
    camtoworlds = np.einsum("nij,ki->nkj", camtoworlds, matrix).astype(np.float32)
    scaling = np.linalg.norm(camtoworlds[:, 0, :3], axis=1)
    camtoworlds[:, :3, :3] = camtoworlds[:, :3, :3] / scaling[:, None, None]
    return camtoworlds


def normalize_scene(camtoworlds: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(t1, camtoworlds)
    points = transform_points(t1, points)

    t2 = align_principal_axes(points)
    camtoworlds = transform_cameras(t2, camtoworlds)
    points = transform_points(t2, points)
    transform = t2 @ t1

    if np.median(points[:, 2]) > np.mean(points[:, 2]):
        t3 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camtoworlds = transform_cameras(t3, camtoworlds)
        points = transform_points(t3, points)
        transform = t3 @ transform

    return camtoworlds, points, transform.astype(np.float32)


def load_colmap_scene(
    data_dir: Path,
    factor: int,
    width: int,
    height: int,
    test_every: int = 8,
    normalize_world_space: bool = True,
) -> ColmapScene:
    try:
        import pycolmap
    except ImportError as exc:
        raise ImportError("Loading 360/COLMAP data requires pycolmap in the active environment.") from exc

    data_dir = Path(data_dir)
    reconstruction = pycolmap.Reconstruction(str(_colmap_dir(data_dir)))

    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    w2c_mats = []
    for image in images:
        w2c_3x4 = np.asarray(image.cam_from_world().matrix(), dtype=np.float32)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :] = w2c_3x4
        w2c_mats.append(w2c)
    camtoworlds = np.linalg.inv(np.stack(w2c_mats, axis=0)).astype(np.float32)

    points = np.asarray([point.xyz for point in reconstruction.points3D.values()], dtype=np.float32)
    colors = (np.asarray([point.color for point in reconstruction.points3D.values()], dtype=np.float32) / 255.0).clip(0.0, 1.0)
    raw_point_count = int(points.shape[0])

    transform = np.eye(4, dtype=np.float32)
    if normalize_world_space:
        camtoworlds, points, transform = normalize_scene(camtoworlds, points)

    image_dir = data_dir / ("images" if factor <= 1 else f"images_{factor}")
    colmap_image_dir = data_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder {image_dir} does not exist")
    if not colmap_image_dir.exists():
        raise FileNotFoundError(f"COLMAP image folder {colmap_image_dir} does not exist")

    colmap_files = sorted(_get_rel_paths(colmap_image_dir))
    image_files = sorted(_get_rel_paths(image_dir))
    if len(colmap_files) != len(image_files):
        raise ValueError(
            f"Image count mismatch: {colmap_image_dir} has {len(colmap_files)}, "
            f"{image_dir} has {len(image_files)}"
        )
    colmap_to_image = dict(zip(colmap_files, image_files))

    training_cameras = []
    for index, (image, camtoworld) in enumerate(zip(images, camtoworlds, strict=True)):
        camera = reconstruction.cameras[image.camera_id]
        if image.name not in colmap_to_image:
            raise FileNotFoundError(f"Image {image.name} not found under {image_dir}")
        image_path = image_dir / colmap_to_image[image.name]
        with Image.open(image_path) as pil_image:
            raw_width, raw_height = pil_image.size

        k = np.asarray(camera.calibration_matrix(), dtype=np.float32)
        k[0, :] *= float(raw_width) / float(camera.width)
        k[1, :] *= float(raw_height) / float(camera.height)
        k[0, :] *= float(width) / float(raw_width)
        k[1, :] *= float(height) / float(raw_height)

        training_cameras.append(
            ColmapTrainingCamera(
                index=index,
                viewmat=np.linalg.inv(camtoworld).astype(np.float32),
                K=k,
                image_path=image_path,
                raw_width=raw_width,
                raw_height=raw_height,
                camera_id=int(image.camera_id),
                image_name=image.name,
            )
        )

    camera_locations = camtoworlds[:, :3, 3]
    scene_center = np.mean(camera_locations, axis=0)
    scene_scale = float(np.max(np.linalg.norm(camera_locations - scene_center, axis=1)))
    return ColmapScene(
        cameras=training_cameras,
        points=points.astype(np.float32),
        colors=colors.astype(np.float32),
        raw_point_count=raw_point_count,
        scene_scale=scene_scale,
        transform=transform,
        data_factor=int(factor),
        test_every=int(test_every),
        normalize_world_space=bool(normalize_world_space),
    )


def select_colmap_cameras(
    cameras: list[ColmapTrainingCamera],
    split: str,
    test_every: int,
    max_frames: int,
    frame_step: int,
    start_index: int,
) -> list[ColmapTrainingCamera]:
    if split not in {"train", "val", "all"}:
        raise ValueError(f"Unsupported COLMAP split {split}")
    selected = []
    for camera in cameras:
        if camera.index < start_index:
            continue
        if split == "train" and camera.index % test_every == 0:
            continue
        if split == "val" and camera.index % test_every != 0:
            continue
        selected.append(camera)
    if frame_step > 1:
        selected = selected[::frame_step]
    if max_frames > 0:
        selected = selected[:max_frames]
    if not selected:
        raise RuntimeError(f"No COLMAP cameras selected for split={split}")
    return selected


def prepare_colmap_points(
    scene: ColmapScene,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    points = scene.points
    colors = scene.colors
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]
    return points.astype(np.float32), colors.astype(np.float32), scene.raw_point_count
