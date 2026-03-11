#!/usr/bin/env python3
"""
Client script for the RBM eval server. No robometer dependency.

Sends a video (or .npy/.npz frames) and task instruction to a running eval server,
then saves per-frame progress and success predictions plus an optional plot.

Example:
  # Start the server first (in another terminal):
  #   uv run python robometer/evals/eval_server.py --config_path=robometer/configs/config.yaml --host=0.0.0.0 --port=8000

  python scripts/example_inference.py --eval-server-url http://localhost:8000 --video /path/to/video.mp4 --task "Pick up the red block"
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import requests
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def create_combined_progress_success_plot(
    progress_pred: np.ndarray,
    num_frames: int,
    success_binary: Optional[np.ndarray] = None,
    success_probs: Optional[np.ndarray] = None,
    success_labels: Optional[np.ndarray] = None,
    is_discrete_mode: bool = False,
    title: Optional[str] = None,
    loss: Optional[float] = None,
    pearson: Optional[float] = None,
) -> Any:
    """Create a combined plot with progress, success binary, and success probabilities.

    This function creates a unified plot with 1 subplot (progress only) or 3 subplots
    (progress, success binary, success probs), similar to the one used in compile_results.py.

    Args:
        progress_pred: Progress predictions array
        num_frames: Number of frames
        success_binary: Optional binary success predictions
        success_probs: Optional success probability predictions
        success_labels: Optional ground truth success labels
        is_discrete_mode: Whether progress is in discrete mode (deprecated, kept for compatibility)
        title: Optional title for the plot (if None, auto-generated from loss/pearson)
        loss: Optional loss value to display in title
        pearson: Optional pearson correlation to display in title

    Returns:
        matplotlib Figure object
    """
    # Determine if we should show success plots
    has_success_binary = success_binary is not None and len(success_binary) == len(progress_pred)

    if has_success_binary:
        # Three subplots: progress, success (binary), success_probs
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.5))
        ax = axs[0]  # Progress subplot
        ax2 = axs[1]  # Success subplot (binary)
        ax3 = axs[2]  # Success probs subplot
    else:
        # Single subplot: progress only
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax2 = None
        ax3 = None

    # Plot progress
    ax.plot(progress_pred, linewidth=2)
    ax.set_ylabel("Progress")

    # Build title
    if title is None:
        title_parts = ["Progress"]
        if loss is not None:
            title_parts.append(f"Loss: {loss:.3f}")
        if pearson is not None:
            title_parts.append(f"Pearson: {pearson:.2f}")
        title = ", ".join(title_parts)
    fig.suptitle(title)

    # Set y-limits and ticks (always continuous since discrete is converted before this function)
    ax.set_ylim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ax.set_yticks(y_ticks)

    # Setup success binary subplot
    if ax2 is not None:
        ax2.step(range(len(success_binary)), success_binary, where="post", linewidth=2, label="Predicted", color="blue")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_binary):
            ax2.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
            )
        ax2.set_ylabel("Success (Binary)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.set_yticks([0, 1])
        ax2.legend()

    # Setup success probs subplot if available
    if ax3 is not None and success_probs is not None:
        ax3.plot(range(len(success_probs)), success_probs, linewidth=2, label="Success Prob", color="purple")
        # Add ground truth success labels as green line if available
        if success_labels is not None and len(success_labels) == len(success_probs):
            ax3.step(
                range(len(success_labels)),
                success_labels,
                where="post",
                linewidth=2,
                label="Ground Truth",
                color="green",
                linestyle="--",
            )
        ax3.set_ylabel("Success Probability")
        ax3.set_ylim(-0.05, 1.05)
        ax3.spines["right"].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax3.legend()

    plt.tight_layout()
    return fig


def create_synchronized_visualization_video(
    frames: np.ndarray,
    progress_pred: np.ndarray,
    out_video_path: str,
    fps: float,
    success_probs: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> str:
    """
    Create a vertically stacked video:
      1) Original robot frame
      2) Progress curve (up to current frame)
      3) Success probability curve (up to current frame)
    """
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape (T,H,W,C), got {frames.shape}")

    num_frames = int(frames.shape[0])
    if num_frames == 0:
        raise ValueError("Cannot create visualization video from empty frame array.")

    if progress_pred.size == 0:
        raise ValueError("Cannot create visualization video with empty progress predictions.")

    has_success_probs = success_probs is not None and success_probs.size > 0
    if has_success_probs:
        T = min(num_frames, int(progress_pred.shape[0]), int(success_probs.shape[0]))
    else:
        T = min(num_frames, int(progress_pred.shape[0]))
    if T <= 0:
        raise ValueError("No overlapping timesteps between frames and predictions.")

    frames = frames[:T]
    progress_pred = progress_pred[:T]
    success_probs_use = success_probs[:T] if has_success_probs else None

    height, width = int(frames.shape[1]), int(frames.shape[2])
    plot_panel_height = max(180, height // 2)
    plots_height = 2 * plot_panel_height
    dpi = 100
    fig_w = width / dpi
    fig_h = plots_height / dpi

    fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=dpi, sharex=True)
    ax_progress, ax_success = axs

    canvas = FigureCanvasAgg(fig)
    x = np.arange(T)

    # Progress subplot.
    (progress_line,) = ax_progress.plot([], [], color="tab:blue", linewidth=2, label="Progress")
    progress_cursor = ax_progress.axvline(0, color="tab:red", linestyle="--", linewidth=1.5, label="Current Frame")
    ax_progress.set_ylabel("Progress")
    ax_progress.set_ylim(-0.05, 1.05)
    ax_progress.set_xlim(0, max(T - 1, 1))
    ax_progress.spines["right"].set_visible(False)
    ax_progress.spines["top"].set_visible(False)
    ax_progress.legend(loc="upper left")

    # Success probability subplot.
    if success_probs_use is not None:
        (success_line,) = ax_success.plot([], [], color="purple", linewidth=2, label="Success Prob")
        success_cursor = ax_success.axvline(0, color="tab:red", linestyle="--", linewidth=1.5, label="Current Frame")
        ax_success.legend(loc="upper left")
    else:
        success_line = None
        success_cursor = None
        ax_success.text(
            0.02,
            0.5,
            "Success probability not available",
            transform=ax_success.transAxes,
            fontsize=10,
            va="center",
        )
    ax_success.set_ylabel("Success Prob")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_xlim(0, max(T - 1, 1))
    ax_success.spines["right"].set_visible(False)
    ax_success.spines["top"].set_visible(False)
    ax_success.set_xlabel("Frame")

    fig.suptitle(title or "Synchronized Progress/Success Visualization")
    fig.tight_layout()

    out_video = Path(out_video_path)
    out_video.parent.mkdir(parents=True, exist_ok=True)
    output_fps = float(fps) if fps and fps > 0 else 1.0

    # yuv420p requires even dimensions for broad player compatibility.
    final_width = width if width % 2 == 0 else width + 1
    final_height = (height + plots_height) if (height + plots_height) % 2 == 0 else (height + plots_height + 1)

    writer = imageio.get_writer(
        str(out_video),
        fps=output_fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    try:
        for t in range(T):
            progress_line.set_data(x[: t + 1], progress_pred[: t + 1])
            progress_cursor.set_xdata([t, t])
            if success_line is not None and success_probs_use is not None and success_cursor is not None:
                success_line.set_data(x[: t + 1], success_probs_use[: t + 1])
                success_cursor.set_xdata([t, t])

            canvas.draw()
            plot_rgba = np.asarray(canvas.buffer_rgba())
            plot_rgb = plot_rgba[:, :, :3]

            frame_rgb = frames[t]
            if frame_rgb.ndim == 2:
                frame_rgb = np.stack([frame_rgb] * 3, axis=-1)
            if frame_rgb.shape[-1] == 1:
                frame_rgb = np.repeat(frame_rgb, repeats=3, axis=-1)
            if frame_rgb.dtype != np.uint8:
                frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

            if plot_rgb.shape[1] != width or plot_rgb.shape[0] != plots_height:
                plot_rgb = np.array(Image.fromarray(plot_rgb).resize((width, plots_height), Image.Resampling.BILINEAR))

            combined_rgb = np.concatenate([frame_rgb, plot_rgb], axis=0)
            if combined_rgb.shape[1] != final_width or combined_rgb.shape[0] != final_height:
                padded = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                padded[: combined_rgb.shape[0], : combined_rgb.shape[1], :] = combined_rgb
                combined_rgb = padded
            writer.append_data(combined_rgb)
    finally:
        writer.close()
        plt.close(fig)

    return str(out_video)


def extract_frames(video_path: str, fps: float = 1.0) -> np.ndarray:
    """Extract frames from video file as numpy array (T, H, W, C).

    Supports both local file paths and URLs (e.g., HuggingFace Hub URLs).
    Uses the provided ``fps`` to control how densely frames are sampled from
    the underlying video; there is no additional hard cap on the number of frames.

    Args:
        video_path: Path to video file or URL
        fps: Frames per second to extract (default: 1.0)

    Returns:
        numpy array of shape (T, H, W, C) containing extracted frames, or None if error
    """
    if video_path is None:
        raise ValueError("video_path is None")

    if isinstance(video_path, tuple):
        video_path = video_path[0]

    # Check if it's a URL or local file
    is_url = video_path.startswith(("http://", "https://"))
    is_local_file = os.path.exists(video_path) if not is_url else False

    if not is_url and not is_local_file:
        raise FileNotFoundError(video_path)

    try:
        import decord  # type: ignore

        # decord.VideoReader can handle both local files and URLs
        vr = decord.VideoReader(video_path, num_threads=1)
        total_frames = len(vr)

        # Determine native FPS; fall back to a reasonable default if unavailable
        try:
            native_fps = float(vr.get_avg_fps())
        except Exception:
            native_fps = 1.0

        # If user-specified fps is invalid or None, default to native fps
        if fps is None or fps <= 0:
            fps = native_fps

        # Compute how many frames we want based on desired fps
        # num_frames ≈ total_duration * fps = total_frames * (fps / native_fps)
        if native_fps > 0:
            desired_frames = int(round(total_frames * (fps / native_fps)))
        else:
            desired_frames = total_frames

        # Clamp to [1, total_frames]
        desired_frames = max(1, min(desired_frames, total_frames))

        # Evenly sample indices to match the desired number of frames
        if desired_frames == total_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, desired_frames, dtype=int).tolist()

        frames_array = vr.get_batch(frame_indices).asnumpy()  # Shape: (T, H, W, C)
        del vr
        return frames_array
    except Exception as e:
        raise RuntimeError(f"Error extracting frames from {video_path}: {e}")


def load_frames_input(
    video_or_array_path: str,
    *,
    fps: float = 1.0,
) -> np.ndarray:
    """
    Load frames from a video file (path or URL) or from a .npy/.npz file.

    Video: uses decord; fps controls sampling density.
    .npy/.npz: expects array shape (T, H, W, C) or (T, C, H, W); for .npz uses
    key 'arr_0' or the first array.

    Returns:
        Frames as uint8 array (T, H, W, C). Raises on failure.
    """
    if video_or_array_path.endswith(".npy"):
        frames_array = np.load(video_or_array_path)
    elif video_or_array_path.endswith(".npz"):
        with np.load(video_or_array_path, allow_pickle=False) as npz:
            if "arr_0" in npz:
                frames_array = npz["arr_0"].copy()
            else:
                frames_array = next(iter(npz.values())).copy()
    else:
        frames_array = extract_frames(video_or_array_path, fps=fps)

    if frames_array is None or frames_array.size == 0:
        raise RuntimeError("Could not extract frames from input.")

    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    if frames_array.ndim == 4:
        if frames_array.shape[1] in (1, 3) and frames_array.shape[-1] not in (1, 3):
            frames_array = frames_array.transpose(0, 2, 3, 1)

    return frames_array


def _numpy_to_npy_file_tuple(arr: np.ndarray, filename: str) -> Tuple[str, io.BytesIO, str]:
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return (filename, buf, "application/octet-stream")


def build_multipart_payload(samples: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Convert a list of sample dicts into:
      - files: mapping for requests.post(files=...)
      - data: mapping for requests.post(data=...) with sample_{i} JSON strings

    Numpy arrays inside trajectory fields are moved to .npy blobs and replaced by
    {"__numpy_file__": <file_key>} references.
    """
    files: Dict[str, Any] = {}
    data: Dict[str, str] = {}

    numpy_fields = ["frames", "lang_vector", "video_embeddings"]

    for i, sample in enumerate(samples):
        sample_copy = json.loads(json.dumps(sample, default=str))  # make JSON-serializable shell
        traj = sample.get("trajectory", {})
        traj_copy = sample_copy.get("trajectory", {})

        for field in numpy_fields:
            val = traj.get(field, None)
            if val is None:
                continue

            # torch.Tensor -> numpy (if torch is available)
            if hasattr(val, "detach") and hasattr(val, "cpu"):
                val = val.detach().cpu().numpy()

            if isinstance(val, np.ndarray):
                file_key = f"sample_{i}_trajectory_{field}"
                files[file_key] = _numpy_to_npy_file_tuple(val, f"{file_key}.npy")
                traj_copy[field] = {"__numpy_file__": file_key}
            else:
                traj_copy[field] = val

        # Keep a helpful frames_shape (as list of ints) if present
        if "frames_shape" in traj_copy and isinstance(traj_copy["frames_shape"], (tuple, list)):
            traj_copy["frames_shape"] = [int(x) for x in traj_copy["frames_shape"]]

        sample_copy["trajectory"] = traj_copy
        data[f"sample_{i}"] = json.dumps(sample_copy)

    return files, data


def post_evaluate_batch_npy(
    eval_server_url: str,
    samples: List[Dict[str, Any]],
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Dict[str, Any]:
    files, data = build_multipart_payload(samples)
    data["use_frame_steps"] = "true" if use_frame_steps else "false"
    url = eval_server_url.rstrip("/") + "/evaluate_batch_npy"
    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def extract_rewards_from_server_output(outputs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse server JSON into per-frame progress and success probability arrays.

    Returns:
        progress_array: Per-frame progress (reward) predictions for the first sample.
        success_array: Per-frame success probabilities, or empty array if not in response.
    """
    outputs_progress = outputs.get("outputs_progress")
    if outputs_progress is None:
        raise ValueError("No `outputs_progress` in server response")
    progress_pred = outputs_progress.get("progress_pred", [])

    if progress_pred and len(progress_pred) > 0:
        progress_array = np.array(progress_pred[0], dtype=np.float32)
    else:
        progress_array = np.array([], dtype=np.float32)

    outputs_success = outputs.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    if success_probs and len(success_probs) > 0:
        success_array = np.array(success_probs[0], dtype=np.float32)
    else:
        success_array = np.array([], dtype=np.float32)

    return progress_array, success_array


def make_progress_sample(
    frames: np.ndarray,
    task: str,
    sample_id: str,
    subsequence_length: int,
) -> Dict[str, Any]:
    return {
        "sample_type": "progress",
        "trajectory": {
            "frames": frames,
            "frames_shape": tuple(frames.shape),
            "task": task,
            "id": sample_id,
            "metadata": {"subsequence_length": int(subsequence_length)},
            "video_embeddings": None,
        },
    }


def compute_rewards_per_frame(
    eval_server_url: str,
    video_frames: np.ndarray,
    task: str,
    timeout_s: float = 120.0,
    use_frame_steps: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Send the full trajectory to the eval server and get per-frame progress and success.

    Args:
        use_frame_steps: If True, server expands into frame-step sub-samples (0:1, 0:2, ...)
            and aggregates; can improve alignment with training. If False, one forward pass on
            the full trajectory (subsampled to fixed frames on the server).

    Returns:
        progress: Per-frame progress (reward) predictions.
        success_probs: Per-frame success probabilities (empty if model has no success head).
    """
    T = int(video_frames.shape[0])
    sample = make_progress_sample(
        frames=video_frames,
        task=task,
        sample_id="0",
        subsequence_length=T,
    )
    outputs = post_evaluate_batch_npy(
        eval_server_url, [sample], timeout_s=timeout_s, use_frame_steps=use_frame_steps
    )
    return extract_rewards_from_server_output(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get per-frame progress and success predictions from an RBM eval server.",
        epilog="Start the server with: uv run python robometer/evals/eval_server.py --config_path=robometer/configs/config.yaml --host=0.0.0.0 --port=8000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--eval-server-url",
        type=str,
        default="http://localhost:8000",
        help="Eval server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path or URL to a video, or a .npy/.npz file with frames (T,H,W,C) or (T,C,H,W)",
    )
    parser.add_argument("--task", type=str, required=True, help="Task instruction describing the trajectory")
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Frames per second when sampling from video (default: 1.0)",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0, help="HTTP request timeout in seconds (default: 120)")
    parser.add_argument(
        "--use-frame-steps",
        action="store_true",
        help="If set, server uses frame-step expansion (0:1, 0:2, ...) and aggregates; can improve reward alignment",
    )
    parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.5,
        help="Threshold for binary success curve in the plot (default: 0.5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for rewards .npy (default: <video_stem>_rewards.npy)",
    )
    parser.add_argument(
        "--out-video",
        type=str,
        default=None,
        help="Output path for synchronized visualization .mp4 (default: <video_stem>_rewards_progress_success.mp4)",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.out) if args.out is not None else video_path.with_name(video_path.stem + "_rewards.npy")

    frames = load_frames_input(str(args.video), fps=float(args.fps))

    rewards, success_probs = compute_rewards_per_frame(
        eval_server_url=args.eval_server_url,
        video_frames=frames,
        task=args.task,
        timeout_s=float(args.timeout_s),
        use_frame_steps=args.use_frame_steps,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), rewards)
    success_path = out_path.with_name(out_path.stem + "_success_probs.npy")
    np.save(str(success_path), success_probs)

    show_success = success_probs.size > 0 and success_probs.size == rewards.size
    success_binary = (success_probs > float(args.success_threshold)).astype(np.int32) if show_success else None
    fig = create_combined_progress_success_plot(
        progress_pred=rewards,
        num_frames=int(frames.shape[0]),
        success_binary=success_binary,
        success_probs=success_probs if show_success else None,
        success_labels=None,
        title=f"Progress/Success — {video_path.name}",
    )
    plot_path = out_path.with_name(out_path.stem + "_progress_success.png")
    fig.savefig(str(plot_path), dpi=200)
    plt.close(fig)

    viz_video_path = Path(args.out_video) if args.out_video is not None else out_path.with_name(
        out_path.stem + "_progress_success.mp4"
    )
    try:
        out_viz_video = create_synchronized_visualization_video(
            frames=frames,
            progress_pred=rewards,
            success_probs=success_probs if show_success else None,
            out_video_path=str(viz_video_path),
            fps=float(args.fps),
            title=f"Progress/Success Over Time — {video_path.name}",
        )
    except Exception as e:
        out_viz_video = None
        print(f"Warning: failed to create synchronized visualization video: {e}")

    summary = {
        "video": str(video_path),
        "num_frames": int(frames.shape[0]),
        "out_npy": str(out_path),
        "out_success_probs_npy": str(success_path),
        "out_plot_png": str(plot_path),
        "out_visualization_mp4": out_viz_video,
        "reward_min": float(np.min(rewards)) if rewards.size else None,
        "reward_max": float(np.max(rewards)) if rewards.size else None,
        "reward_mean": float(np.mean(rewards)) if rewards.size else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
