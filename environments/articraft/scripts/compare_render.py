"""Render multi-view comparison between dataset references and RL outputs.

Usage (on KAOLA debug pod):
    python compare_render.py --output-dir /tmp/articraft-compare

Produces per-record PNG images with:
  - Top row: dataset reference (4 viewpoints)
  - Bottom row: RL output (4 viewpoints)
  - Title: prompt text, reward, compile status

Requirements: trimesh, pyrender, pillow, numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = "s3://arcwm-code-us-west-2/ericzyma"
LATEST_EXPERIMENT = "articraft-0602-phase2-v3sync"
DATASET_LOCAL = Path("/local-ssd/data/articraft/records")
ARTICRAFT_SOURCE = Path("/data/work/prime-rl/environments/articraft/source")

# Camera viewpoints: (azimuth_deg, elevation_deg, label)
VIEWPOINTS = [
    (225, 30, "3/4 view"),
    (180, 0, "front"),
    (270, 0, "right"),
    (180, 90, "top"),
]

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    success: bool
    urdf_xml: str | None = None
    scene: Any = None  # trimesh.Scene
    error: str | None = None
    asset_root: Path | None = None


@dataclass
class ComparisonRecord:
    record_id: str
    prompt_text: str
    category_slug: str
    reference_model_py: str
    rl_model_py: str
    rl_reward: float | None = None
    rl_turns: int = 0
    rl_trajectory_id: str = ""


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def s3_cp(s3_path: str) -> str | None:
    """Download file from S3 and return contents as string."""
    try:
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except Exception:
        return None


def s3_ls(s3_path: str) -> list[str]:
    """List S3 prefix, return lines."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []
    except Exception:
        return []


def s3_ls_recursive(s3_path: str) -> list[str]:
    """List S3 prefix recursively, return lines."""
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", "--recursive", s3_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")
        return []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Compile model.py -> trimesh Scene
# ---------------------------------------------------------------------------
def compile_model_to_scene(model_py_code: str, work_dir: Path) -> CompileResult:
    """Execute model.py code and export to a trimesh Scene for rendering."""
    import trimesh

    # Write model.py to work_dir
    script_path = work_dir / "model.py"
    script_path.write_text(model_py_code, encoding="utf-8")

    # Setup SDK path
    if str(ARTICRAFT_SOURCE) not in sys.path:
        sys.path.insert(0, str(ARTICRAFT_SOURCE))

    try:
        # Activate asset session
        from sdk._core.v0.assets import AssetSession, activate_asset_session
        session = AssetSession.ephemeral()
        activate_asset_session(session)

        # Execute the model script
        globals_dict: dict[str, Any] = {"__file__": str(script_path)}
        exec(compile(model_py_code, str(script_path), "exec"), globals_dict)

        object_model = globals_dict.get("object_model")
        if object_model is None:
            return CompileResult(success=False, error="No object_model found in script")

        # Export to URDF
        from sdk._core.v0._urdf_export import compile_object_to_urdf_xml
        urdf_xml = compile_object_to_urdf_xml(
            object_model,
            asset_root=session.asset_root,
            include_physical_collisions=False,
            validate=False,
        )

        # Build trimesh scene from URDF
        scene = _urdf_to_trimesh_scene(urdf_xml, session.asset_root)

        return CompileResult(
            success=True,
            urdf_xml=urdf_xml,
            scene=scene,
            asset_root=session.asset_root,
        )

    except Exception as exc:
        return CompileResult(
            success=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}",
        )


def _urdf_to_trimesh_scene(urdf_xml: str, asset_root: Path) -> Any:
    """Parse URDF XML and build a trimesh Scene with colored meshes."""
    import trimesh
    import xml.etree.ElementTree as ET

    root = ET.fromstring(urdf_xml)
    scene = trimesh.Scene()

    # Parse materials
    materials: dict[str, tuple[float, ...]] = {}
    for mat_el in root.findall("material"):
        name = mat_el.attrib.get("name", "")
        color_el = mat_el.find("color")
        if color_el is not None:
            rgba_str = color_el.attrib.get("rgba", "0.5 0.5 0.5 1.0")
            rgba = tuple(float(x) for x in rgba_str.split())
            materials[name] = rgba

    # Parse each link's visuals
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "unknown")
        for i, visual in enumerate(link.findall("visual")):
            geom = visual.find("geometry")
            if geom is None:
                continue

            mesh = None
            origin = visual.find("origin")
            xyz = (0.0, 0.0, 0.0)
            rpy = (0.0, 0.0, 0.0)
            if origin is not None:
                xyz_str = origin.attrib.get("xyz", "0 0 0")
                rpy_str = origin.attrib.get("rpy", "0 0 0")
                xyz = tuple(float(x) for x in xyz_str.split())
                rpy = tuple(float(x) for x in rpy_str.split())

            # Determine geometry type
            box = geom.find("box")
            cylinder = geom.find("cylinder")
            sphere = geom.find("sphere")
            mesh_el = geom.find("mesh")

            if box is not None:
                size = tuple(float(x) for x in box.attrib.get("size", "0.1 0.1 0.1").split())
                mesh = trimesh.creation.box(extents=size)
            elif cylinder is not None:
                radius = float(cylinder.attrib.get("radius", "0.05"))
                length = float(cylinder.attrib.get("length", "0.1"))
                mesh = trimesh.creation.cylinder(radius=radius, height=length)
            elif sphere is not None:
                radius = float(sphere.attrib.get("radius", "0.05"))
                mesh = trimesh.creation.icosphere(radius=radius)
            elif mesh_el is not None:
                filename = mesh_el.attrib.get("filename", "")
                mesh_path = asset_root / filename
                if mesh_path.exists():
                    try:
                        mesh = trimesh.load_mesh(str(mesh_path), force="mesh")
                    except Exception:
                        continue
                else:
                    continue

            if mesh is None:
                continue

            # Apply transform
            transform = _rpy_xyz_to_matrix(rpy, xyz)
            mesh.apply_transform(transform)

            # Apply color
            mat_el_vis = visual.find("material")
            rgba = (0.5, 0.5, 0.5, 1.0)
            if mat_el_vis is not None:
                mat_name = mat_el_vis.attrib.get("name", "")
                if mat_name in materials:
                    rgba = materials[mat_name]
                color_el = mat_el_vis.find("color")
                if color_el is not None:
                    rgba = tuple(float(x) for x in color_el.attrib.get("rgba", "0.5 0.5 0.5 1.0").split())

            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                face_colors=np.array([int(c * 255) for c in rgba[:4]], dtype=np.uint8),
            )

            scene.add_geometry(mesh, node_name=f"{link_name}_visual_{i}")

    return scene


def _rpy_xyz_to_matrix(rpy: tuple, xyz: tuple) -> np.ndarray:
    """Convert RPY euler angles + XYZ translation to 4x4 transform matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    # ZYX rotation order (URDF convention)
    rot = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])

    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = xyz
    return mat


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_scene_multiview(scene, viewpoints: list[tuple] = VIEWPOINTS) -> list[np.ndarray]:
    """Render a trimesh scene from multiple viewpoints. Returns list of RGB arrays."""
    import pyrender

    images = []

    if scene is None or len(scene.geometry) == 0:
        for _ in viewpoints:
            img = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 200, dtype=np.uint8)
            images.append(img)
        return images

    for azimuth, elevation, label in viewpoints:
        pr_scene = pyrender.Scene(bg_color=[240, 240, 240, 255])

        # Add meshes
        for name, geom in scene.geometry.items():
            try:
                pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
            except Exception:
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom)
                except Exception:
                    continue

            node_transform = np.eye(4)
            try:
                node_transform = scene.graph.get(name)[0]
            except Exception:
                pass
            pr_scene.add(pr_mesh, pose=node_transform)

        # Compute camera position from scene bounds
        bounds = scene.bounds
        center = (bounds[0] + bounds[1]) / 2.0
        extent = np.linalg.norm(bounds[1] - bounds[0])
        distance = extent * 1.8

        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        cam_x = center[0] + distance * np.cos(el_rad) * np.cos(az_rad)
        cam_y = center[1] + distance * np.cos(el_rad) * np.sin(az_rad)
        cam_z = center[2] + distance * np.sin(el_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        camera_pose = _look_at(cam_pos, center, up=np.array([0, 0, 1]))
        camera = pyrender.PerspectiveCamera(yfov=np.radians(45))
        pr_scene.add(camera, pose=camera_pose)

        # Lights
        light = pyrender.DirectionalLight(color=[255, 255, 255], intensity=3.0)
        pr_scene.add(light, pose=camera_pose)
        ambient = pyrender.DirectionalLight(color=[255, 255, 255], intensity=1.5)
        ambient_pose = np.eye(4)
        ambient_pose[:3, 3] = center + np.array([0, 0, distance])
        pr_scene.add(ambient, pose=ambient_pose)

        try:
            renderer = pyrender.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
            color, _ = renderer.render(pr_scene)
            renderer.delete()
            images.append(color)
        except Exception as exc:
            logger.warning(f"Render failed for viewpoint {label}: {exc}")
            images.append(np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 200, dtype=np.uint8))

    return images


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Compute a 4x4 camera pose matrix (OpenGL convention: -Z forward)."""
    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


# ---------------------------------------------------------------------------
# Compose comparison image
# ---------------------------------------------------------------------------

def compose_comparison(
    ref_images: list[np.ndarray],
    rl_images: list[np.ndarray],
    record: ComparisonRecord,
    ref_compile: CompileResult,
    rl_compile: CompileResult,
    output_path: Path,
) -> None:
    """Compose a side-by-side comparison image and save as PNG."""
    from PIL import Image, ImageDraw, ImageFont

    n_views = len(VIEWPOINTS)
    margin = 10
    header_height = 100
    label_height = 30
    total_width = n_views * IMAGE_WIDTH + (n_views + 1) * margin
    total_height = header_height + 2 * (label_height + IMAGE_HEIGHT) + 3 * margin

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Header
    prompt_short = record.prompt_text[:120] + ("..." if len(record.prompt_text) > 120 else "")
    draw.text((margin, 8), f"Category: {record.category_slug}", fill=(0, 0, 0), font=font_large)
    draw.text((margin, 30), f"Prompt: {prompt_short}", fill=(80, 80, 80), font=font_small)
    draw.text((margin, 50), f"Record: {record.record_id[:50]}", fill=(120, 120, 120), font=font_small)
    reward_str = f"{record.rl_reward:.3f}" if record.rl_reward is not None else "N/A"
    draw.text((margin, 70), f"RL reward: {reward_str} | turns: {record.rl_turns}", fill=(0, 0, 180), font=font_small)

    # Reference row
    y_ref = header_height
    ref_status = "compiled OK" if ref_compile.success else f"FAILED: {(ref_compile.error or '')[:60]}"
    draw.text((margin, y_ref + 5), f"REFERENCE (dataset): {ref_status}", fill=(0, 128, 0), font=font_small)
    y_ref += label_height

    # RL row
    y_rl = y_ref + IMAGE_HEIGHT + margin
    rl_status = "compiled OK" if rl_compile.success else f"FAILED: {(rl_compile.error or '')[:60]}"
    draw.text((margin, y_rl + 5), f"RL OUTPUT: {rl_status}", fill=(180, 0, 0), font=font_small)
    y_rl += label_height

    for i in range(n_views):
        x = margin + i * (IMAGE_WIDTH + margin)
        ref_img = Image.fromarray(ref_images[i]) if i < len(ref_images) else Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (200, 200, 200))
        canvas.paste(ref_img, (x, y_ref))
        rl_img = Image.fromarray(rl_images[i]) if i < len(rl_images) else Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), (200, 200, 200))
        canvas.paste(rl_img, (x, y_rl))
        _, _, vp_label = VIEWPOINTS[i]
        draw.text((x + 5, y_ref + IMAGE_HEIGHT - 20), vp_label, fill=(255, 255, 255), font=font_small)
        draw.text((x + 5, y_rl + IMAGE_HEIGHT - 20), vp_label, fill=(255, 255, 255), font=font_small)

    canvas.save(output_path, "PNG")
    logger.info(f"Saved comparison: {output_path}")


# ---------------------------------------------------------------------------
# Find matching pairs
# ---------------------------------------------------------------------------
def find_eval_pairs() -> list[ComparisonRecord]:
    """Find matching pairs of dataset reference + RL output for eval examples."""
    records = []

    s3_eval_prefix = f"{S3_BUCKET}/experiments/{LATEST_EXPERIMENT}/output/articraft-work/eval/"
    lines = s3_ls(s3_eval_prefix)

    for line in lines:
        line = line.strip()
        if "PRE" not in line:
            continue
        dir_name = line.split("PRE")[-1].strip().rstrip("/")
        parts = dir_name.split("__", 1)
        if len(parts) < 2:
            continue
        record_id_prefix = parts[1]

        # Find the full record_id from local dataset
        matching_dirs = list(DATASET_LOCAL.glob(f"{record_id_prefix}*"))
        if not matching_dirs:
            logger.warning(f"No local record found for prefix: {record_id_prefix}")
            continue
        record_dir = matching_dirs[0]
        record_id = record_dir.name

        # Read reference model.py
        ref_model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        if not ref_model_path.exists():
            logger.warning(f"No reference model.py at {ref_model_path}")
            continue
        ref_model_py = ref_model_path.read_text(encoding="utf-8")

        # Read prompt
        prompt_path = record_dir / "revisions" / "rev_000001" / "prompt.txt"
        prompt_text = ""
        if prompt_path.exists():
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()

        # Find latest RL trajectory with best reward
        traj_prefix = f"{s3_eval_prefix}{dir_name}/"
        traj_lines = s3_ls_recursive(traj_prefix)
        meta_paths = [l for l in traj_lines if "meta.json" in l]

        best_reward = -999.0
        best_traj_id = ""
        best_meta: dict = {}
        for meta_line in meta_paths:
            parts_path = meta_line.strip().split()
            if len(parts_path) < 4:
                continue
            s3_key = parts_path[3]
            meta_content = s3_cp(f"s3://arcwm-code-us-west-2/{s3_key}")
            if meta_content:
                try:
                    meta = json.loads(meta_content)
                    reward = meta.get("final_reward") or 0.0
                    if reward > best_reward:
                        best_reward = reward
                        best_traj_id = s3_key.split("/")[-2]
                        best_meta = meta
                except json.JSONDecodeError:
                    continue

        if not best_traj_id:
            logger.warning(f"No RL trajectories found for {dir_name}")
            continue

        # Download RL model.py
        rl_model_py = s3_cp(
            f"s3://arcwm-code-us-west-2/ericzyma/experiments/{LATEST_EXPERIMENT}"
            f"/output/articraft-work/eval/{dir_name}/{best_traj_id}/model.py"
        )
        if not rl_model_py:
            logger.warning(f"Could not download RL model.py for {dir_name}/{best_traj_id}")
            continue

        # Read category from record.json
        record_json_path = record_dir / "record.json"
        category_slug = ""
        if record_json_path.exists():
            try:
                rj = json.loads(record_json_path.read_text())
                category_slug = rj.get("category_slug", "")
            except Exception:
                pass

        records.append(ComparisonRecord(
            record_id=record_id,
            prompt_text=prompt_text,
            category_slug=category_slug,
            reference_model_py=ref_model_py,
            rl_model_py=rl_model_py,
            rl_reward=best_reward,
            rl_turns=best_meta.get("turns_used", 0),
            rl_trajectory_id=best_traj_id,
        ))

    return records


def find_train_pairs(max_samples: int = 5) -> list[ComparisonRecord]:
    """Find matching pairs from train outputs (diverse categories)."""
    records = []
    s3_train_prefix = f"{S3_BUCKET}/experiments/{LATEST_EXPERIMENT}/output/articraft-work/train/"
    lines = s3_ls(s3_train_prefix)

    seen_categories: set[str] = set()

    for line in lines:
        if len(records) >= max_samples:
            break
        line = line.strip()
        if "PRE" not in line:
            continue
        dir_name = line.split("PRE")[-1].strip().rstrip("/")
        parts = dir_name.split("__", 1)
        if len(parts) < 2:
            continue
        record_id_prefix = parts[1]

        # Find local record
        matching_dirs = list(DATASET_LOCAL.glob(f"{record_id_prefix}*"))
        if not matching_dirs:
            continue
        record_dir = matching_dirs[0]
        record_id = record_dir.name

        # Check category
        record_json_path = record_dir / "record.json"
        category_slug = ""
        if record_json_path.exists():
            try:
                rj = json.loads(record_json_path.read_text())
                category_slug = rj.get("category_slug", "")
            except Exception:
                pass

        if category_slug in seen_categories:
            continue

        # Read reference
        ref_model_path = record_dir / "revisions" / "rev_000001" / "model.py"
        if not ref_model_path.exists():
            continue
        ref_model_py = ref_model_path.read_text(encoding="utf-8")

        # Read prompt
        prompt_path = record_dir / "revisions" / "rev_000001" / "prompt.txt"
        prompt_text = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.exists() else ""

        # Find best RL trajectory
        traj_prefix = f"{s3_train_prefix}{dir_name}/"
        traj_lines = s3_ls_recursive(traj_prefix)
        meta_paths = [l for l in traj_lines if "meta.json" in l]

        best_reward = -999.0
        best_traj_id = ""
        best_meta: dict = {}
        for meta_line in meta_paths[:8]:
            parts_path = meta_line.strip().split()
            if len(parts_path) < 4:
                continue
            s3_key = parts_path[3]
            meta_content = s3_cp(f"s3://arcwm-code-us-west-2/{s3_key}")
            if meta_content:
                try:
                    meta = json.loads(meta_content)
                    reward = meta.get("final_reward") or 0.0
                    if reward > best_reward:
                        best_reward = reward
                        best_traj_id = s3_key.split("/")[-2]
                        best_meta = meta
                except json.JSONDecodeError:
                    continue

        if not best_traj_id:
            continue

        rl_model_py = s3_cp(
            f"s3://arcwm-code-us-west-2/ericzyma/experiments/{LATEST_EXPERIMENT}"
            f"/output/articraft-work/train/{dir_name}/{best_traj_id}/model.py"
        )
        if not rl_model_py:
            continue

        seen_categories.add(category_slug)
        records.append(ComparisonRecord(
            record_id=record_id,
            prompt_text=prompt_text,
            category_slug=category_slug,
            reference_model_py=ref_model_py,
            rl_model_py=rl_model_py,
            rl_reward=best_reward,
            rl_turns=best_meta.get("turns_used", 0),
            rl_trajectory_id=best_traj_id,
        ))
        logger.info(f"Found train pair: {category_slug} (reward={best_reward:.3f})")

    return records


# ---------------------------------------------------------------------------
# Output filename
# ---------------------------------------------------------------------------

def _build_output_filename(index: int, n_eval: int, record: ComparisonRecord) -> str:
    """Build a PNG filename that surfaces split / reward / sample id at a glance.

    Sorted listing groups by split first, then orders by index within split.
    Reward suffix lets a human eyeball trends without opening summary.json.
    """
    split = "eval" if index < n_eval else "train"
    reward_tag = f"r{record.rl_reward:.2f}" if record.rl_reward is not None else "rNA"
    # record_id is already category_slug + hash — unique and traceable back to dataset
    sample_id = record.record_id[:60]
    return f"{split}_{index:03d}_{reward_tag}_{sample_id}.png"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render comparison: dataset vs RL output")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/articraft-compare"))
    parser.add_argument("--eval-only", action="store_true", help="Only compare eval examples")
    parser.add_argument("--train-samples", type=int, default=5)
    parser.add_argument("--upload-s3", action="store_true", help="Upload results to S3")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set pyrender to use EGL for headless rendering
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    logger.info("Finding eval pairs...")
    pairs = find_eval_pairs()
    logger.info(f"Found {len(pairs)} eval pairs")
    n_eval = len(pairs)

    if not args.eval_only:
        logger.info("Finding train pairs...")
        train_pairs = find_train_pairs(max_samples=args.train_samples)
        logger.info(f"Found {len(train_pairs)} train pairs")
        pairs.extend(train_pairs)

    if not pairs:
        logger.error("No comparison pairs found!")
        return

    summary = []
    for i, record in enumerate(pairs):
        logger.info(f"\n[{i+1}/{len(pairs)}] {record.category_slug} ({record.record_id[:30]}...)")

        with tempfile.TemporaryDirectory(prefix="ref_") as ref_dir, \
             tempfile.TemporaryDirectory(prefix="rl_") as rl_dir:

            logger.info("  Compiling reference...")
            ref_result = compile_model_to_scene(record.reference_model_py, Path(ref_dir))
            if not ref_result.success:
                logger.warning(f"  Reference compile failed: {(ref_result.error or '')[:100]}")

            logger.info("  Compiling RL output...")
            rl_result = compile_model_to_scene(record.rl_model_py, Path(rl_dir))
            if not rl_result.success:
                logger.warning(f"  RL compile failed: {(rl_result.error or '')[:100]}")

            logger.info("  Rendering...")
            ref_images = render_scene_multiview(ref_result.scene if ref_result.success else None)
            rl_images = render_scene_multiview(rl_result.scene if rl_result.success else None)

            output_path = (
                args.output_dir
                / _build_output_filename(i, n_eval, record)
            )
            compose_comparison(ref_images, rl_images, record, ref_result, rl_result, output_path)

            summary.append({
                "index": i,
                "category": record.category_slug,
                "record_id": record.record_id,
                "rl_reward": record.rl_reward,
                "rl_turns": record.rl_turns,
                "ref_compiled": ref_result.success,
                "rl_compiled": rl_result.success,
                "output_file": output_path.name,
            })

    # Write summary
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"\nDone! {len(pairs)} comparisons -> {args.output_dir}")

    if args.upload_s3:
        s3_dest = f"{S3_BUCKET}/experiments/{LATEST_EXPERIMENT}/compare/"
        logger.info(f"Uploading to {s3_dest}...")
        subprocess.run(["aws", "s3", "sync", str(args.output_dir), s3_dest], check=True)
        logger.info("Upload complete!")


if __name__ == "__main__":
    main()