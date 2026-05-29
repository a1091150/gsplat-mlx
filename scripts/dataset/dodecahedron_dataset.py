from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from training_dataset import TrainingCamera, TrainingDataset, image_to_u8, write_png


BG = np.array([0.025, 0.025, 0.025], dtype=np.float32)


def dodecahedron_geometry() -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) * 0.5
    normals = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            normals.append([0.0, s1, s2 * phi])
            normals.append([s1, s2 * phi, 0.0])
            normals.append([s1 * phi, 0.0, s2])
    normals = np.asarray(normals, dtype=np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    vertices = []
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            for k in range(j + 1, len(normals)):
                mat = np.stack([normals[i], normals[j], normals[k]], axis=0)
                if abs(np.linalg.det(mat)) < 1.0e-8:
                    continue
                point = np.linalg.solve(mat, np.ones((3,), dtype=np.float64))
                if np.all(normals @ point <= 1.0 + 1.0e-6):
                    vertices.append(point)
    verts = []
    for point in vertices:
        if not any(np.linalg.norm(point - prev) < 1.0e-5 for prev in verts):
            verts.append(point)
    vertices_np = np.asarray(verts, dtype=np.float64)
    vertices_np /= np.max(np.linalg.norm(vertices_np, axis=1))

    faces: list[list[int]] = []
    for normal in normals:
        face = np.where(np.abs(vertices_np @ normal - np.max(vertices_np @ normal)) < 1.0e-5)[0]
        center = vertices_np[face].mean(axis=0)
        axis_u = vertices_np[face[0]] - center
        axis_u /= np.linalg.norm(axis_u)
        axis_v = np.cross(normal, axis_u)
        angles = np.arctan2((vertices_np[face] - center) @ axis_v, (vertices_np[face] - center) @ axis_u)
        faces.append(face[np.argsort(angles)].astype(int).tolist())
    return vertices_np.astype(np.float32), faces, normals.astype(np.float32)


def face_colors(count: int) -> np.ndarray:
    start = np.array([1.0, 0.04, 0.02], dtype=np.float32)
    mid = np.array([0.06, 0.74, 0.95], dtype=np.float32)
    end = np.array([0.62, 0.08, 0.90], dtype=np.float32)
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    first = (1.0 - np.minimum(t * 2.0, 1.0)) * start + np.minimum(t * 2.0, 1.0) * mid
    second = (1.0 - np.maximum((t - 0.5) * 2.0, 0.0)) * mid + np.maximum((t - 0.5) * 2.0, 0.0) * end
    return np.where(t <= 0.5, first, second).astype(np.float32)


def look_at_viewmat(position: np.ndarray, target: np.ndarray = np.zeros(3, dtype=np.float32)) -> np.ndarray:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_guess))) > 0.95:
        up_guess = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(up_guess, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    rot = np.stack([right, up, forward], axis=0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rot
    view[:3, 3] = -rot @ position
    return view


def make_camera_positions(count: int, radius: float) -> np.ndarray:
    positions = []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(count):
        y = 1.0 - 2.0 * (i + 0.5) / float(count)
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        positions.append([radius * r * np.cos(theta), radius * y, radius * r * np.sin(theta)])
    return np.asarray(positions, dtype=np.float32)


def make_intrinsics(width: int, height: int, focal_scale: float) -> np.ndarray:
    focal = focal_scale * float(min(width, height))
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def project_vertices(vertices: np.ndarray, viewmat: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    verts_h = np.concatenate([vertices, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (viewmat @ verts_h.T).T[:, :3]
    z = np.clip(cam[:, 2], 1.0e-5, None)
    uv = np.empty((vertices.shape[0], 2), dtype=np.float32)
    uv[:, 0] = K[0, 0] * cam[:, 0] / z + K[0, 2]
    uv[:, 1] = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return uv, cam


def render_dodecahedron(
    vertices: np.ndarray,
    faces: list[list[int]],
    normals: np.ndarray,
    colors: np.ndarray,
    viewmat: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    uv, cam = project_vertices(vertices, viewmat, K)
    image = Image.new("RGB", (width, height), tuple((BG * 255).astype(np.uint8).tolist()))
    draw = ImageDraw.Draw(image)
    rot = viewmat[:3, :3]
    visible_faces = []
    for face_id, face in enumerate(faces):
        if np.any(cam[face, 2] <= 0.01):
            continue
        normal_cam = rot @ normals[face_id]
        center_cam = cam[face].mean(axis=0)
        if float(np.dot(normal_cam, center_cam)) < 0.0:
            visible_faces.append(face_id)
    face_order = sorted(visible_faces, key=lambda idx: float(np.mean(cam[faces[idx], 2])), reverse=True)
    for face_id in face_order:
        face = faces[face_id]
        pts = [(float(uv[idx, 0]), float(uv[idx, 1])) for idx in face]
        color = tuple((np.clip(colors[face_id], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8).tolist())
        draw.polygon(pts, fill=color, outline=(255, 255, 255))
    line_width = max(1, int(round(min(width, height) / 192.0)))
    for face_id in face_order:
        face = faces[face_id]
        pts = [(float(uv[idx, 0]), float(uv[idx, 1])) for idx in face]
        draw.line(pts + [pts[0]], fill=(255, 255, 255), width=line_width)
    return np.asarray(image, dtype=np.float32) / 255.0


def sample_dodecahedron_foreground_points(
    vertices: np.ndarray,
    faces: list[list[int]],
    colors: np.ndarray,
    samples_per_face: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    points = []
    point_colors = []
    for face_id, face in enumerate(faces):
        face_vertices = vertices[np.asarray(face, dtype=np.int32)]
        center = face_vertices.mean(axis=0)
        triangles = [
            (center, face_vertices[idx], face_vertices[(idx + 1) % len(face_vertices)])
            for idx in range(len(face_vertices))
        ]
        per_triangle = max(1, int(np.ceil(samples_per_face / len(triangles))))
        for a, b, c in triangles:
            uv = rng.random((per_triangle, 2), dtype=np.float32)
            sqrt_u = np.sqrt(uv[:, 0:1])
            tri_points = (1.0 - sqrt_u) * a + sqrt_u * (1.0 - uv[:, 1:2]) * b + sqrt_u * uv[:, 1:2] * c
            points.append(tri_points.astype(np.float32))
            point_colors.append(np.broadcast_to(colors[face_id], tri_points.shape).astype(np.float32))
    return np.concatenate(points, axis=0), np.concatenate(point_colors, axis=0)


def load_dodecahedron_dataset(
    out_dir: Path,
    width: int,
    height: int,
    camera_count: int,
    radius: float,
    focal_scale: float,
) -> TrainingDataset:
    out_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, normals = dodecahedron_geometry()
    colors = face_colors(len(faces))
    foreground_points, foreground_colors = sample_dodecahedron_foreground_points(vertices, faces, colors)
    K = make_intrinsics(width, height, focal_scale)
    cameras = []
    metadata = {
        "width": width,
        "height": height,
        "camera_count": camera_count,
        "camera_radius": radius,
        "focal_scale": focal_scale,
        "vertices": vertices.astype(float).tolist(),
        "faces": faces,
        "face_colors": colors.astype(float).tolist(),
        "foreground_points": int(foreground_points.shape[0]),
        "frames": [],
    }
    for idx, position in enumerate(make_camera_positions(camera_count, radius)):
        viewmat = look_at_viewmat(position)
        target = render_dodecahedron(vertices, faces, normals, colors, viewmat, K, width, height)
        image_path = out_dir / f"frame_{idx:03d}.png"
        write_png(image_path, image_to_u8(target))
        cameras.append(TrainingCamera(idx, viewmat, K.copy(), position, target))
        metadata["frames"].append(
            {
                "index": idx,
                "image_path": str(image_path),
                "position": position.astype(float).tolist(),
                "viewmat": viewmat.astype(float).tolist(),
                "K": K.astype(float).tolist(),
            }
        )
    (out_dir / "dataset_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return TrainingDataset(
        name="dodecahedron",
        cameras=cameras,
        bbox_min=vertices.min(axis=0).astype(np.float32),
        bbox_max=vertices.max(axis=0).astype(np.float32),
        metadata=metadata,
        foreground_points=foreground_points,
        foreground_colors=foreground_colors,
        background_color=BG.copy(),
    )
