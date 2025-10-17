"""Capture multi-view images and end-effector trajectory for a ManiSkill task."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import gymnasium as gym
import numpy as np
import sapien
import torch
from PIL import Image
from transforms3d import quaternions
import tyro

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import Camera, CameraConfig
from mani_skill.utils import sapien_utils


# Predefined camera viewpoints expressed as (eye, target) pairs.
CAMERA_VIEWS = (
    ("front", [0.6, 0.0, 0.45], [0.0, 0.0, 0.2]),
    ("top", [0.05, 0.0, 1.0], [0.0, 0.0, 0.1]),
    ("left", [0.2, 0.6, 0.45], [0.0, 0.0, 0.2]),
    ("right", [0.2, -0.6, 0.45], [0.0, 0.0, 0.2]),
    ("diagonal", [0.55, 0.35, 0.6], [0.0, 0.0, 0.2]),
)


@dataclass
class Args:
    """CLI arguments parsed by Tyro."""

    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: Annotated[str, tyro.conf.arg(aliases=["-o"])] = "none"
    control_mode: Annotated[Optional[str], tyro.conf.arg(aliases=["-c"])] = None
    shader: str = "default"  # Options: "minimal", "default", "rt-fast", "rt"
    sim_backend: Annotated[str, tyro.conf.arg(aliases=["-b"])] = "auto"
    render_backend: str = "gpu"  # Options: "gpu", "cpu", "auto"
    robot_uids: Annotated[Optional[str], tyro.conf.arg(aliases=["-r"])] = None
    seed: Annotated[Optional[int], tyro.conf.arg(aliases=["-s"])] = None
    max_steps: Annotated[int, tyro.conf.arg(aliases=["-m"])] = 50
    output_root: str = "outputs"
    image_width: int = 640
    image_height: int = 480
    hide_robot: Annotated[bool, tyro.conf.arg(aliases=["--hide"])] = False
    show_gripper_marker: Annotated[bool, tyro.conf.arg(aliases=["--marker"])] = False
    gripper_width: float = 0.05  # Width of the gripper visual (distance between fingers)


def _create_gripper_visual(scene, grasp_width: float = 0.05):
    """Create a visual representation of the gripper pose (from ManiSkill motion planning).

    This creates a semi-transparent gripper visualization showing:
    - Center sphere (blue)
    - Palm/base (green box)
    - Two finger indicators (blue and red boxes)

    Args:
        scene: The ManiSkillScene
        grasp_width: Width of the gripper opening

    Returns:
        The created gripper visual actor (kinematic, no collision)
    """
    builder = scene.create_actor_builder()
    grasp_pose_visual_width = 0.01

    # Center sphere
    builder.add_sphere_visual(
        pose=sapien.Pose(p=[0, 0, 0.0]),
        radius=grasp_pose_visual_width,
        material=sapien.render.RenderMaterial(base_color=[0.3, 0.4, 0.8, 0.7])
    )

    # Palm base (vertical green box)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.08]),
        half_size=[grasp_pose_visual_width, grasp_pose_visual_width, 0.02],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )

    # Palm horizontal bar (green)
    builder.add_box_visual(
        pose=sapien.Pose(p=[0, 0, -0.05]),
        half_size=[grasp_pose_visual_width, grasp_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 1, 0, 0.7]),
    )

    # Left finger (blue)
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                grasp_width + grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[0, 0, 1, 0.7]),
    )

    # Right finger (red)
    builder.add_box_visual(
        pose=sapien.Pose(
            p=[
                0.03 - grasp_pose_visual_width * 3,
                -grasp_width - grasp_pose_visual_width,
                0.03 - 0.05,
            ],
            q=quaternions.axangle2quat(np.array([0, 1, 0]), theta=np.pi / 2),
        ),
        half_size=[0.04, grasp_pose_visual_width, grasp_pose_visual_width],
        material=sapien.render.RenderMaterial(base_color=[1, 0, 0, 0.7]),
    )

    builder.set_initial_pose(sapien.Pose())
    return builder.build_kinematic(name="gripper_visual")


def _build_camera_bundle(shader: str, width: int = 640, height: int = 480) -> Dict[str, CameraConfig]:
    configs: Dict[str, CameraConfig] = {}
    for uid, eye, target in CAMERA_VIEWS:
        pose = sapien_utils.look_at(eye=eye, target=target)
        configs[uid] = CameraConfig(
            uid=uid,
            pose=pose,
            width=width,
            height=height,
            fov=np.deg2rad(55.0),
            near=0.01,
            far=3.0,
            shader_pack=shader,
        )
    return configs


def _attach_human_cameras(scene, configs: Dict[str, CameraConfig]) -> Dict[str, Camera]:
    sensors: Dict[str, Camera] = {}
    for uid, config in configs.items():
        sensor = Camera(config, scene)
        sensors[uid] = sensor
        scene.human_render_cameras[uid] = sensor
    if scene.gpu_sim_enabled and hasattr(scene, "_human_render_cameras_initialized"):
        scene._human_render_cameras_initialized = False
    return sensors


def _capture_images(cameras: Dict[str, Camera], scene, destination: Path, step: int) -> None:
    scene.update_render(update_sensors=False, update_human_render_cameras=True)
    step_dir = destination / "images" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    for uid, camera in cameras.items():
        camera.capture()
        obs = camera.get_obs(rgb=True, depth=False, position=False, segmentation=False)
        rgb = obs["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype in (np.float32, np.float64):
            rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        Image.fromarray(np.ascontiguousarray(rgb)).save(step_dir / f"{uid}.png")


def _append_trajectory(record: List[dict], agent, step: int) -> None:
    if agent is None:
        return
    tcp_pose = agent.tcp_pose
    pos = tcp_pose.p
    quat = tcp_pose.q
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(quat, torch.Tensor):
        quat = quat.detach().cpu().numpy()
    if pos.ndim > 1:
        pos = pos[0]
    if quat.ndim > 1:
        quat = quat[0]
    qpos = agent.robot.get_qpos()
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu().numpy()
    if qpos.ndim > 1:
        qpos = qpos[0]
    record.append({
        "step": step,
        "tcp_position": pos.tolist(),
        "tcp_quaternion": quat.tolist(),
        "joint_positions": qpos.tolist(),
    })


def _write_trajectory(record: List[dict], destination: Path) -> None:
    (destination / "trajectory").mkdir(exist_ok=True)
    json_path = destination / "trajectory" / "trajectory.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    npz_path = destination / "trajectory" / "trajectory.npz"
    if record:
        np.savez(
            npz_path,
            steps=np.array([item["step"] for item in record], dtype=np.int32),
            tcp_position=np.array([item["tcp_position"] for item in record], dtype=np.float32),
            tcp_quaternion=np.array([item["tcp_quaternion"] for item in record], dtype=np.float32),
            joint_positions=np.array([item["joint_positions"] for item in record], dtype=np.float32),
        )
    else:
        np.savez(npz_path, steps=np.array([], dtype=np.int32))


def main(args: Args) -> None:
    if args.seed is not None:
        np.random.seed(args.seed)

    env_kwargs = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=None,
        num_envs=1,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        enable_shadow=False,  # Disable shadows to avoid robot shadow artifacts
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )
    if args.robot_uids:
        robots = tuple(r.strip() for r in args.robot_uids.split(",") if r.strip())
        env_kwargs["robot_uids"] = robots[0] if len(robots) == 1 else robots

    env: BaseEnv = gym.make(args.env_id, **env_kwargs)
    if args.seed is not None and env.action_space is not None:
        env.action_space.seed(args.seed)

    env.reset(seed=args.seed, options=dict(reconfigure=True))
    cameras = _attach_human_cameras(
        env.unwrapped.scene,
        _build_camera_bundle(args.shader, args.image_width, args.image_height)
    )

    # Hide robot and/or create gripper visual if requested
    gripper_visual = None
    if args.hide_robot and env.unwrapped.agent is not None:
        # Hide all links of the robot by setting visibility to 0
        for link in env.unwrapped.agent.robot.links:
            for obj in link._objs:
                rb_comp = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
                if rb_comp is not None:
                    rb_comp.visibility = 0
        print("Robot hidden from view.")

    if args.show_gripper_marker and env.unwrapped.agent is not None:
        gripper_visual = _create_gripper_visual(env.unwrapped.scene, grasp_width=args.gripper_width)
        print(f"Gripper visual created (width={args.gripper_width}).")

    run_root = (Path(__file__).resolve().parent / args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True, exist_ok=True)

    trajectory: List[dict] = []
    print(f"Starting capture for {args.max_steps} steps. Press Ctrl+C to stop early.")
    print(f"Shader: {args.shader}, Resolution: {args.image_width}x{args.image_height}")
    if args.hide_robot:
        print("Mode: Robot hidden")
    if args.show_gripper_marker:
        print("Mode: Gripper marker visible")

    try:
        for step in range(args.max_steps):
            action = env.action_space.sample() if env.action_space is not None else None
            _, _, terminated, truncated, _ = env.step(action)

            # Update gripper visual position if it exists
            if gripper_visual is not None and env.unwrapped.agent is not None:
                tcp_pose = env.unwrapped.agent.tcp.pose
                gripper_visual.set_pose(tcp_pose)

            _capture_images(cameras, env.unwrapped.scene, run_root, step)
            _append_trajectory(trajectory, env.unwrapped.agent, step)

            if (terminated | truncated).any():
                env.reset()

            if (step + 1) % 50 == 0:
                print(f"Progress: {step + 1}/{args.max_steps} steps captured")

    except KeyboardInterrupt:
        print("\nCapture interrupted by user.")

    finally:
        _write_trajectory(trajectory, run_root)
        env.close()
        print("Recording finished.")
        print(f"Saved {len(trajectory)} steps")
        print(f"Images: {run_root / 'images'}")
        print(f"Trajectory: {run_root / 'trajectory'}")


if __name__ == "__main__":
    main(tyro.cli(Args))