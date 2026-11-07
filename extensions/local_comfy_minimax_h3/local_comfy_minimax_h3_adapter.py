"""Local-only MiniMax H3 channel for an already-running loopback ComfyUI.

This adapter never calls MiniMax, Hailuo, RunningHub, Comfy Cloud, or any
other paid API. It submits a native ComfyUI graph to a loopback server and
copies the saved MP4 back to the requested local path.

FL2VA (text, first frame, first+last frame) follows the ComfyUI 0.43.0
blueprint ``Image to Video (MiniMax H3)`` at commit
ab08f7b2caa6d4007ecde3762116c6193613f43b:

- ``MiniMaxH3ImageToVideo``
- FL2VA int8 diffusion model
- optional ``minimax_h3_fl2v_turbo_8step`` LoRA at strength 1
- ``BasicGuider`` (the blueprint does not use CFG)
- ``KSamplerSelect`` ``res_multistep``
- ``BasicScheduler`` ``simple``
- video VAE fp16 + audio VAE fp32
- ``VAEDecode`` + ``VAEDecodeAudio`` + ``CreateVideo`` at 24 fps + ``SaveVideo``

Reference-to-video is a separate explicit mode. It uses
``MiniMaxH3ReferenceToVideo`` and the Ref2VA weights. It is never substituted
for ``first_last_frame``.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

try:
    from custom_tools.audio_generation.base_tool_compat import BaseTool
    from custom_tools.video_generation.output_dir_utils import (
        default_video_output_dir,
        resolve_video_output_dir,
    )
except ImportError:  # pragma: no cover - package layout during a direct file run
    BaseTool = object  # type: ignore[misc,assignment]
    default_video_output_dir = None  # type: ignore[assignment]
    resolve_video_output_dir = None  # type: ignore[assignment]


ENGINE = "local-comfy-minimax-h3"
PROVIDER = "local_comfyui"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
BLOCKED_PORT = 7860
DEFAULT_SEED = 5555
FPS = 24
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
SMOKE_MIN_EDGE = 256
SMOKE_FRAMES = 22
TURBO_STEPS = 8
QUALITY_STEPS = 20
REF_TURBO_STEPS = 4
MIN_MP4_BYTES = 4096

FL2VA_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
FL2VA_LORA = "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
REF2VA_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
REF2VA_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"

ASPECT_RATIOS = {
    "9:16": (9, 16),
    "16:9": (16, 9),
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "21:9": (21, 9),
}

FL2VA_MODES = {"text_to_video", "image_to_video", "first_last_frame"}
REFERENCE_MODES = {"reference_to_video", "ref2va"}
PROFILES = {"smoke", "turbo", "quality"}
PAID_ENGINES = {
    "seedance2.0",
    "minimax-h3",
    "runninghub-minimax-h3-fl2va",
    "runninghub-minimax-h3-multi-reference",
    "runninghub-minimax-h3-text-to-video",
    "agnes-video-v2.0",
}

ALLOWED_CLASS_TYPES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
    "LoadImage",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
    "BasicGuider",
    "BasicScheduler",
    "KSamplerSelect",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
}

CLOUD_CLASS_MARKERS = ("comfycloud", "hailuo", "minimaxtexttovideo", "minimaximagetovideo")

FL2VA_NODES = (
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "LoadImage",
    "MiniMaxH3ImageToVideo",
    "BasicGuider",
    "BasicScheduler",
    "KSamplerSelect",
    "RandomNoise",
    "SamplerCustomAdvanced",
    "VAEDecode",
    "VAEDecodeAudio",
    "CreateVideo",
    "SaveVideo",
    "LoraLoaderModelOnly",
)


class LocalH3Error(RuntimeError):
    """Fail-closed error. Callers must not fall back to a cloud engine."""


class LocalComfyMiniMaxH3Schema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str = Field(..., description="Video prompt")
    output_dir: str = Field("", description="Directory under the capsule output root")
    output_path: Optional[str] = Field(None, description="Exact local mp4 path")
    generation_type: str = Field("first_last_frame")
    engine: str = Field(ENGINE)
    image_path: Optional[str] = None
    start_image_path: Optional[str] = None
    end_image_path: Optional[str] = None
    first_frame_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    image_paths: Optional[list[str]] = None
    images: Optional[list[str]] = None
    aspect_ratio: str = Field("9:16")
    duration: Optional[float] = None
    seed: Optional[int] = None
    profile: Optional[str] = None
    comfy_url: Optional[str] = None
    dry_run: bool = False


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def comfy_url_from_env() -> str:
    return _env("CAPSULE_LOCAL_COMFY_URL", DEFAULT_COMFY_URL) or DEFAULT_COMFY_URL


def profile_from_env(explicit: Optional[str] = None) -> str:
    value = (explicit or _env("CAPSULE_LOCAL_H3_PROFILE", "turbo") or "turbo").strip().lower()
    if value not in PROFILES:
        raise LocalH3Error(
            f"unsupported local H3 profile {value!r}; expected one of {sorted(PROFILES)}"
        )
    return value


def timeout_seconds(profile: str, explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = _env("CAPSULE_LOCAL_H3_TIMEOUT_SECONDS")
    if raw:
        return float(raw)
    return 1800.0 if profile == "smoke" else 3600.0


def poll_seconds() -> float:
    raw = _env("CAPSULE_LOCAL_H3_POLL_SECONDS", "3") or "3"
    return max(0.2, float(raw))


def assert_loopback_url(url: str) -> str:
    """Accept only a loopback ComfyUI origin. Port 7860 is the music service."""
    parsed = urllib.parse.urlparse(url.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not host:
        raise LocalH3Error(f"unsafe ComfyUI endpoint {url!r}; expected http://127.0.0.1:8188")
    if parsed.username or parsed.password:
        raise LocalH3Error("unsafe ComfyUI endpoint includes credentials")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise LocalH3Error(
            f"unsafe ComfyUI endpoint host {host!r}; only loopback is allowed, no cloud fallback"
        )
    lowered = url.lower()
    for marker in ("minimaxi.com", "runninghub", "comfy.org", "cloud.comfy"):
        if marker in lowered:
            raise LocalH3Error(f"unsafe ComfyUI endpoint contains {marker}")
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    if port == BLOCKED_PORT:
        raise LocalH3Error(
            "port 7860 is the ACE-Step music service; refusing to submit a video prompt"
        )
    path = parsed.path or ""
    if path not in {"", "/"}:
        raise LocalH3Error("ComfyUI endpoint must be the server root, not a subpath")
    return f"{parsed.scheme}://{host}:{port}"


def align_frame_count(length: int) -> int:
    """Snap up to the model's 17k+5 grid. Copied from ComfyUI nodes_minimax_h3."""
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
        if frame_count > 3600:
            raise LocalH3Error("requested frame count exceeds the local H3 node maximum")
    return frame_count


def frames_for_seconds(duration: float) -> int:
    """Official blueprint expression: max(5, round(seconds * 24)) snapped to 17k+5."""
    if duration <= 0:
        raise LocalH3Error("duration must be positive")
    return align_frame_count(int(round(float(duration) * FPS)))


def adapt_canvas(width: float, height: float) -> tuple[int, int]:
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32."""
    if width <= 0 or height <= 0:
        raise LocalH3Error("canvas width and height must be positive")
    ratio = width / height
    if ratio >= 1.0:
        nominal_w, nominal_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nominal_w, nominal_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nominal_w * nominal_h > MAX_PIXELS:
        scale = (MAX_PIXELS / (nominal_w * nominal_h)) ** 0.5
        nominal_w, nominal_h = nominal_w * scale, nominal_h * scale

    def snap(value: float) -> int:
        return max(CANVAS_MULTIPLE, int(round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))

    return snap(nominal_w), snap(nominal_h)


def smoke_canvas(width: int, height: int) -> tuple[int, int]:
    def snap(value: int) -> int:
        return max(SMOKE_MIN_EDGE, (int(value) // 2) // CANVAS_MULTIPLE * CANVAS_MULTIPLE)

    return snap(width), snap(height)


def canvas_for_aspect(aspect_ratio: str, profile: str) -> tuple[int, int]:
    key = (aspect_ratio or "").strip()
    if key == "adaptive":
        raise LocalH3Error("adaptive canvas needs an input image; resolve the ratio before calling this")
    if key not in ASPECT_RATIOS:
        raise LocalH3Error(
            f"unsupported aspect ratio {key!r}; expected one of {sorted(ASPECT_RATIOS)} or adaptive"
        )
    ratio_w, ratio_h = ASPECT_RATIOS[key]
    full = adapt_canvas(ratio_w, ratio_h)
    if profile == "smoke":
        return smoke_canvas(*full)
    return full


def resolve_engine(engine: Optional[str]) -> str:
    value = (engine or ENGINE).strip()
    if value in PAID_ENGINES or value.lower() in {"minimax", "hailuo", "seedance"}:
        raise LocalH3Error(
            f"engine {value!r} is a paid or cloud route; this channel only runs {ENGINE}"
        )
    if value != ENGINE:
        raise LocalH3Error(f"engine {value!r} is not the local ComfyUI H3 channel {ENGINE}")
    return value


def _clean_paths(values: Optional[Sequence[str]]) -> list[str]:
    cleaned = []
    for value in values or []:
        text = str(value or "").strip()
        if text:
            cleaned.append(text)
    return cleaned


def resolve_mode_images(request: Mapping[str, Any]) -> dict[str, Any]:
    """Map a rea-video / UniversalVideoGenerationTool payload onto one local mode."""
    generation_type = str(request.get("generation_type") or "text_to_video").strip()
    start = str(
        request.get("start_image_path")
        or request.get("first_frame_path")
        or request.get("image_path")
        or ""
    ).strip()
    end = str(request.get("end_image_path") or request.get("last_frame_path") or "").strip()
    images = _clean_paths(request.get("image_paths") or request.get("images"))

    if generation_type in FL2VA_MODES:
        if generation_type == "first_last_frame":
            if not start and not end and len(images) >= 2:
                start, end = images[0], images[1]
                images = images[2:]
            if images:
                raise LocalH3Error(
                    "first_last_frame does not accept a reference image list; "
                    "use generation_type reference_to_video for Ref2VA"
                )
            if not start or not end:
                raise LocalH3Error("first_last_frame requires both a first frame and a last frame")
            return {"mode": "first_last_frame", "family": "fl2va", "first": start, "last": end, "refs": []}
        if generation_type == "image_to_video":
            if not start and images:
                start = images[0]
                images = images[1:]
            if end or images:
                raise LocalH3Error(
                    "image_to_video accepts only the first frame; use first_last_frame when a last frame is set"
                )
            if not start:
                raise LocalH3Error("image_to_video requires image_path or start_image_path")
            return {"mode": "image_to_video", "family": "fl2va", "first": start, "last": "", "refs": []}
        if start or end or images:
            raise LocalH3Error("text_to_video does not accept images; use image_to_video or first_last_frame")
        return {"mode": "text_to_video", "family": "fl2va", "first": "", "last": "", "refs": []}

    if generation_type in REFERENCE_MODES:
        if start or end:
            raise LocalH3Error(
                "reference_to_video uses reference images, not first/last frames; "
                "refusing to run them through MiniMaxH3ImageToVideo"
            )
        if not images:
            raise LocalH3Error("reference_to_video requires image_paths")
        if len(images) > 9:
            raise LocalH3Error("reference_to_video accepts at most 9 reference images")
        return {"mode": "reference_to_video", "family": "ref2va", "first": "", "last": "", "refs": images}

    raise LocalH3Error(
        f"unsupported generation_type {generation_type!r}; no cloud fallback. "
        f"Supported: {sorted(FL2VA_MODES | REFERENCE_MODES)}"
    )


def sampling_for(profile: str, family: str) -> dict[str, Any]:
    if family == "fl2va":
        if profile == "quality":
            return {
                "steps": QUALITY_STEPS,
                "sampler": "res_multistep",
                "lora": None,
                "lora_strength": 0.0,
                "unet": FL2VA_UNET,
            }
        return {
            "steps": TURBO_STEPS,
            "sampler": "res_multistep",
            "lora": FL2VA_LORA,
            "lora_strength": 1.0,
            "unet": FL2VA_UNET,
        }
    if family != "ref2va":
        raise LocalH3Error(f"unknown model family {family!r}")
    if profile == "quality":
        return {
            "steps": 40,
            "sampler": "euler",
            "lora": None,
            "lora_strength": 0.0,
            "unet": REF2VA_UNET,
        }
    if profile == "smoke":
        return {
            "steps": TURBO_STEPS,
            "sampler": "euler",
            "lora": None,
            "lora_strength": 0.0,
            "unet": REF2VA_UNET,
        }
    return {
        "steps": REF_TURBO_STEPS,
        "sampler": "euler",
        "lora": REF2VA_LORA,
        "lora_strength": 1.0,
        "unet": REF2VA_UNET,
    }


def _node(class_type: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    if class_type not in ALLOWED_CLASS_TYPES:
        raise LocalH3Error(f"refusing class_type {class_type}")
    lowered = class_type.lower()
    if any(marker in lowered for marker in CLOUD_CLASS_MARKERS):
        raise LocalH3Error(f"refusing cloud class_type {class_type}")
    return {"class_type": class_type, "inputs": dict(inputs)}


def build_graph(
    *,
    mode: str,
    prompt: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int,
    sampler: str,
    unet: str,
    lora: Optional[str],
    lora_strength: float,
    uploaded: Mapping[str, str],
    filename_prefix: str,
) -> dict[str, Any]:
    """Build a native ComfyUI API prompt. Keys are node ids."""
    if not str(prompt).strip():
        raise LocalH3Error("prompt is empty")
    if width % 32 or height % 32:
        raise LocalH3Error(f"width and height must be multiples of 32, got {width}x{height}")
    length = align_frame_count(length)
    graph: dict[str, Any] = {
        "1": _node("UNETLoader", {"unet_name": unet, "weight_dtype": "default"}),
        "2": _node(
            "CLIPLoader",
            {"clip_name": TEXT_ENCODER, "type": "minimax", "device": "default"},
        ),
        "3": _node("VAELoader", {"vae_name": VIDEO_VAE}),
        "4": _node("VAELoader", {"vae_name": AUDIO_VAE}),
    }
    model_ref = ["1", 0]
    if lora:
        graph["5"] = _node(
            "LoraLoaderModelOnly",
            {"model": model_ref, "lora_name": lora, "strength_model": float(lora_strength)},
        )
        model_ref = ["5", 0]

    if mode in FL2VA_MODES:
        cond_inputs: dict[str, Any] = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
        }
        if mode in {"image_to_video", "first_last_frame"}:
            cond_inputs["first_frame"] = ["6", 0]
            graph["6"] = _node("LoadImage", {"image": uploaded["first"]})
        if mode == "first_last_frame":
            cond_inputs["last_frame"] = ["7", 0]
            graph["7"] = _node("LoadImage", {"image": uploaded["last"]})
        graph["8"] = _node("MiniMaxH3ImageToVideo", cond_inputs)
        positive = ["8", 0]
        latent = ["8", 1]
    elif mode == "reference_to_video":
        cond_inputs = {
            "clip": ["2", 0],
            "prompt": prompt,
            "width": int(width),
            "height": int(height),
            "length": int(length),
            "ref_image_size": "match",
            "vae": ["3", 0],
        }
        for index, name in enumerate(uploaded["refs"], start=1):
            node_id = str(20 + index)
            graph[node_id] = _node("LoadImage", {"image": name})
            cond_inputs[f"ref_images.ref_image_{index}"] = [node_id, 0]
        graph["8"] = _node("MiniMaxH3ReferenceToVideo", cond_inputs)
        positive = ["8", 0]
        latent = ["8", 1]
    else:
        raise LocalH3Error(f"cannot build a graph for mode {mode}")

    graph["9"] = _node("BasicGuider", {"model": model_ref, "conditioning": positive})
    graph["10"] = _node(
        "BasicScheduler",
        {"model": model_ref, "scheduler": "simple", "steps": int(steps), "denoise": 1.0},
    )
    graph["11"] = _node("KSamplerSelect", {"sampler_name": sampler})
    graph["12"] = _node("RandomNoise", {"noise_seed": int(seed)})
    graph["13"] = _node(
        "SamplerCustomAdvanced",
        {
            "noise": ["12", 0],
            "guider": ["9", 0],
            "sampler": ["11", 0],
            "sigmas": ["10", 0],
            "latent_image": latent,
        },
    )
    graph["14"] = _node("VAEDecode", {"samples": ["13", 0], "vae": ["3", 0]})
    graph["15"] = _node("VAEDecodeAudio", {"samples": ["13", 0], "vae": ["4", 0]})
    graph["16"] = _node(
        "CreateVideo",
        {
            "images": ["14", 0],
            "fps": float(FPS),
            "audio": ["15", 0],
            "bit_depth": 8,
            "color_space": "sRGB",
        },
    )
    graph["17"] = _node(
        "SaveVideo",
        {
            "video": ["16", 0],
            "filename_prefix": filename_prefix,
            "format": "mp4",
            "format.codec": "h264",
            "format.codec.encoding": "auto",
        },
    )
    assert_graph_contract(graph, mode)
    return graph


def assert_graph_contract(graph: Mapping[str, Any], mode: str) -> None:
    class_types = [node["class_type"] for node in graph.values()]
    unknown = sorted({item for item in class_types if item not in ALLOWED_CLASS_TYPES})
    if unknown:
        raise LocalH3Error(f"graph contains class types outside the local allowlist: {unknown}")
    for node in graph.values():
        blob = json.dumps(node, ensure_ascii=False).lower()
        if "comfycloud" in blob or "hailuo" in blob or "minimaxi.com" in blob or "api_key" in blob:
            raise LocalH3Error("graph contains a cloud marker or credential field")
    joined = " ".join(class_types)
    if mode in FL2VA_MODES:
        if "MiniMaxH3ReferenceToVideo" in joined:
            raise LocalH3Error("first/last or text mode built a Ref2VA graph")
        if "MiniMaxH3ImageToVideo" not in joined:
            raise LocalH3Error("FL2VA graph is missing MiniMaxH3ImageToVideo")
        unets = [node["inputs"].get("unet_name") for node in graph.values() if node["class_type"] == "UNETLoader"]
        if unets != [FL2VA_UNET]:
            raise LocalH3Error(f"FL2VA graph must load {FL2VA_UNET}, got {unets}")
        image_node = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ImageToVideo")
        if mode == "first_last_frame" and ("first_frame" not in image_node["inputs"] or "last_frame" not in image_node["inputs"]):
            raise LocalH3Error("first_last_frame graph is missing first_frame or last_frame")
        if mode == "image_to_video" and ("first_frame" not in image_node["inputs"] or "last_frame" in image_node["inputs"]):
            raise LocalH3Error("image_to_video graph must connect only the first frame")
        if mode == "text_to_video" and ("first_frame" in image_node["inputs"] or "last_frame" in image_node["inputs"]):
            raise LocalH3Error("text_to_video graph must not connect frames")
    elif mode == "reference_to_video":
        if "MiniMaxH3ImageToVideo" in joined:
            raise LocalH3Error("reference mode built an FL2VA graph")
        if "MiniMaxH3ReferenceToVideo" not in joined:
            raise LocalH3Error("reference graph is missing MiniMaxH3ReferenceToVideo")
        unets = [node["inputs"].get("unet_name") for node in graph.values() if node["class_type"] == "UNETLoader"]
        if unets != [REF2VA_UNET]:
            raise LocalH3Error(f"Ref2VA graph must load {REF2VA_UNET}, got {unets}")
        ref_node = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        if not any(key.startswith("ref_images.ref_image_") for key in ref_node["inputs"]):
            raise LocalH3Error("reference graph has no ref_images.ref_image_* inputs")
    else:
        raise LocalH3Error(f"unknown graph mode {mode}")
    if "SaveVideo" not in joined or "CreateVideo" not in joined:
        raise LocalH3Error("graph does not save an mp4")
    if "CFGGuider" in joined or "MiniMaxH3SigmaShift" in joined:
        raise LocalH3Error("graph added guidance or sigma-shift nodes that the local blueprint does not use")


def _combo_values(node_info: Mapping[str, Any], input_name: str) -> list[Any]:
    inputs = node_info.get("input") or {}
    spec = None
    for section in ("required", "optional"):
        section_inputs = inputs.get(section) or {}
        if input_name in section_inputs:
            spec = section_inputs[input_name]
            break
    if not isinstance(spec, list) or not spec:
        return []
    head = spec[0]
    if isinstance(head, list):
        return list(head)
    if head == "COMBO" and isinstance(spec[1], Mapping):
        return list(spec[1].get("options") or [])
    return []


def preflight_object_info(object_info: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Fail closed when the live server is missing a node, sampler, or installed file."""
    required = set(FL2VA_NODES)
    if plan["mode"] == "reference_to_video":
        required.discard("MiniMaxH3ImageToVideo")
        required.add("MiniMaxH3ReferenceToVideo")
    if not plan.get("lora"):
        required.discard("LoraLoaderModelOnly")
    missing = sorted(name for name in required if name not in object_info)
    if missing:
        raise LocalH3Error("ComfyUI is missing local H3 nodes: " + ", ".join(missing))

    checks = [
        ("UNETLoader", "unet_name", plan["unet"]),
        ("CLIPLoader", "clip_name", TEXT_ENCODER),
        ("CLIPLoader", "type", "minimax"),
        ("VAELoader", "vae_name", VIDEO_VAE),
        ("VAELoader", "vae_name", AUDIO_VAE),
        ("KSamplerSelect", "sampler_name", plan["sampler"]),
        ("BasicScheduler", "scheduler", "simple"),
    ]
    if plan.get("lora"):
        checks.append(("LoraLoaderModelOnly", "lora_name", plan["lora"]))
    problems = []
    for node_name, input_name, expected in checks:
        values = _combo_values(object_info[node_name], input_name)
        if expected not in values:
            problems.append(f"{node_name}.{input_name} missing {expected}")
    if problems:
        raise LocalH3Error("installed ComfyUI files do not match this channel: " + "; ".join(problems))


def plan_generation(request: Mapping[str, Any]) -> dict[str, Any]:
    engine = resolve_engine(request.get("engine"))
    profile = profile_from_env(request.get("profile"))
    mode_info = resolve_mode_images(request)
    prompt = str(request.get("prompt") or "").strip()
    if not prompt:
        raise LocalH3Error("prompt is empty")
    aspect = str(request.get("aspect_ratio") or request.get("ratio") or "9:16").strip()
    if aspect == "adaptive":
        raise LocalH3Error(
            "adaptive aspect is not used for this local channel; pass 9:16, 16:9, 1:1, 4:3, 3:4, or 21:9"
        )
    width, height = canvas_for_aspect(aspect, profile)
    duration = request.get("duration")
    if profile == "smoke":
        length = SMOKE_FRAMES
    else:
        length = frames_for_seconds(float(duration) if duration not in {None, ""} else 5)
    sampling = sampling_for(profile, mode_info["family"])
    seed_value = request.get("seed")
    if seed_value in {None, ""}:
        env_seed = _env("CAPSULE_LOCAL_H3_SEED")
        seed_value = env_seed if env_seed else DEFAULT_SEED
    seed = int(seed_value)
    if seed < 0:
        raise LocalH3Error("seed must be >= 0")
    prefix = "local_h3/" + uuid.uuid4().hex[:12]
    plan = {
        "engine": engine,
        "provider": PROVIDER,
        "profile": profile,
        "mode": mode_info["mode"],
        "family": mode_info["family"],
        "prompt": prompt,
        "aspect_ratio": aspect,
        "duration_seconds": None if duration in {None, ""} else float(duration),
        "width": width,
        "height": height,
        "length": length,
        "fps": FPS,
        "seed": seed,
        "filename_prefix": prefix,
        "first": mode_info["first"],
        "last": mode_info["last"],
        "refs": list(mode_info["refs"]),
        **sampling,
    }
    return plan


def graph_from_plan(plan: Mapping[str, Any], uploaded: Mapping[str, Any]) -> dict[str, Any]:
    return build_graph(
        mode=plan["mode"],
        prompt=plan["prompt"],
        width=int(plan["width"]),
        height=int(plan["height"]),
        length=int(plan["length"]),
        steps=int(plan["steps"]),
        seed=int(plan["seed"]),
        sampler=str(plan["sampler"]),
        unet=str(plan["unet"]),
        lora=plan.get("lora"),
        lora_strength=float(plan.get("lora_strength") or 0),
        uploaded=uploaded,
        filename_prefix=str(plan["filename_prefix"]),
    )


def placeholder_uploads(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan["mode"] == "reference_to_video":
        return {"refs": [f"dry_run_ref_{index}.png" for index in range(1, len(plan["refs"]) + 1)]}
    uploaded: dict[str, Any] = {}
    if plan["first"]:
        uploaded["first"] = "dry_run_first.png"
    if plan["last"]:
        uploaded["last"] = "dry_run_last.png"
    return uploaded


def is_valid_mp4(payload: bytes) -> bool:
    if not payload or len(payload) < MIN_MP4_BYTES:
        return False
    if payload[:1] in {b"<", b"{", b"["}:
        return False
    return payload[4:8] == b"ftyp" or b"ftyp" in payload[:32]


class UrllibTransport:
    def request(
        self,
        method: str,
        url: str,
        data: Optional[bytes] = None,
        headers: Optional[Mapping[str, str]] = None,
        timeout: float = 60,
    ) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=data, headers=dict(headers or {}), method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read()
            return int(exc.code), body


class ComfyClient:
    def __init__(self, base_url: str, transport: Optional[UrllibTransport] = None) -> None:
        self.base_url = assert_loopback_url(base_url)
        self.transport = transport or UrllibTransport()

    def _url(self, path: str, query: Optional[Mapping[str, str]] = None) -> str:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _json(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None, timeout: float = 60) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        status, body = self.transport.request(method, self._url(path), data=data, headers=headers, timeout=timeout)
        text = body.decode("utf-8", errors="replace")
        if status >= 400:
            raise LocalH3Error(f"ComfyUI {method} {path} failed with HTTP {status}: {text[:800]}")
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LocalH3Error(f"ComfyUI {path} returned non-JSON: {text[:200]}") from exc

    def object_info(self, class_names: Sequence[str]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for name in class_names:
            payload = self._json("GET", f"/object_info/{urllib.parse.quote(name)}")
            if isinstance(payload, Mapping) and name in payload:
                merged[name] = payload[name]
        return merged

    def upload_image(self, path: str, upload_name: str) -> str:
        source = Path(path)
        if not source.is_file():
            raise LocalH3Error(f"input image does not exist: {source}")
        content = source.read_bytes()
        if not content:
            raise LocalH3Error(f"input image is empty: {source}")
        suffix = source.suffix.lower() if source.suffix else ".png"
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise LocalH3Error(f"unsupported input image type {suffix}")
        filename = f"{upload_name}{suffix}"
        content_type = mimetypes.types_map.get(suffix, "application/octet-stream")
        boundary = "----LocalH3" + uuid.uuid4().hex
        fields = {"type": "input", "overwrite": "true", "subfolder": "local_h3"}
        chunks: list[bytes] = []
        for key, value in fields.items():
            chunks.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )
        chunks.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(content)
        chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        status, body = self.transport.request(
            "POST",
            self._url("/upload/image"),
            data=b"".join(chunks),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            timeout=120,
        )
        text = body.decode("utf-8", errors="replace")
        if status >= 400:
            raise LocalH3Error(f"ComfyUI image upload failed with HTTP {status}: {text[:800]}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LocalH3Error(f"ComfyUI image upload returned non-JSON: {text[:200]}") from exc
        name = str(payload.get("name") or "")
        subfolder = str(payload.get("subfolder") or "").replace("\\", "/").strip("/")
        if not name:
            raise LocalH3Error("ComfyUI image upload did not return a filename")
        return f"{subfolder}/{name}" if subfolder else name

    def submit(self, graph: Mapping[str, Any], client_id: str) -> str:
        payload = self._json(
            "POST",
            "/prompt",
            {"prompt": graph, "client_id": client_id},
            timeout=120,
        )
        node_errors = payload.get("node_errors") or {}
        if node_errors:
            raise LocalH3Error("ComfyUI rejected the local H3 graph: " + json.dumps(node_errors, ensure_ascii=False)[:800])
        prompt_id = str(payload.get("prompt_id") or "")
        if not prompt_id:
            raise LocalH3Error("ComfyUI did not return a prompt_id")
        return prompt_id

    def history(self, prompt_id: str) -> Optional[Mapping[str, Any]]:
        payload = self._json("GET", f"/history/{urllib.parse.quote(prompt_id)}")
        if not isinstance(payload, Mapping):
            return None
        entry = payload.get(prompt_id)
        if isinstance(entry, Mapping):
            return entry
        return None

    def queue(self) -> Mapping[str, Any]:
        payload = self._json("GET", "/queue")
        return payload if isinstance(payload, Mapping) else {}

    def cancel(self, prompt_id: str) -> None:
        try:
            self._json("POST", "/queue", {"delete": [prompt_id]})
        except LocalH3Error:
            pass
        try:
            self._json("POST", "/interrupt", {"prompt_id": prompt_id})
        except LocalH3Error:
            pass

    def download(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        status, body = self.transport.request(
            "GET",
            self._url(
                "/view",
                {"filename": filename, "subfolder": subfolder, "type": folder_type or "output"},
            ),
            timeout=120,
        )
        if status >= 400:
            raise LocalH3Error(f"ComfyUI view failed with HTTP {status}")
        return body


def _status_error(entry: Mapping[str, Any]) -> str:
    status = entry.get("status") or {}
    messages = status.get("messages") or []
    details = []
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, list) and len(item) >= 2 and item[0] == "execution_error":
                data = item[1] if isinstance(item[1], Mapping) else {}
                details.append(
                    str(data.get("exception_type") or "error")
                    + ": "
                    + str(data.get("exception_message") or data)[:500]
                    + f" (node {data.get('node_type') or data.get('node_id') or '?'})"
                )
    if details:
        return "; ".join(details)
    return str(status.get("status_str") or "error")


def _saved_videos(entry: Mapping[str, Any]) -> list[dict[str, str]]:
    found = []
    outputs = entry.get("outputs") or {}
    if not isinstance(outputs, Mapping):
        return found
    for node_out in outputs.values():
        if not isinstance(node_out, Mapping):
            continue
        for key in ("images", "videos", "gifs"):
            items = node_out.get(key) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                filename = str(item.get("filename") or "")
                if filename.lower().endswith(".mp4"):
                    found.append(
                        {
                            "filename": filename,
                            "subfolder": str(item.get("subfolder") or ""),
                            "type": str(item.get("type") or "output"),
                        }
                    )
    return found


def _queue_ids(items: Any) -> set[str]:
    found = set()
    if not isinstance(items, list):
        return found
    for item in items:
        if isinstance(item, list) and len(item) > 1:
            found.add(str(item[1]))
    return found


def poll_until_video(
    client: ComfyClient,
    prompt_id: str,
    timeout: float,
    poll: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, str]:
    deadline = clock() + timeout
    last_error = ""
    while clock() < deadline:
        try:
            queue = client.queue()
            running = _queue_ids(queue.get("queue_running"))
            pending = _queue_ids(queue.get("queue_pending"))
            if prompt_id in running or prompt_id in pending:
                last_error = ""
        except LocalH3Error as exc:
            last_error = str(exc)
        try:
            entry = client.history(prompt_id)
        except LocalH3Error as exc:
            last_error = str(exc)
            entry = None
        if entry:
            status = entry.get("status") or {}
            status_str = str(status.get("status_str") or "")
            if status_str == "error" or (status.get("completed") and status_str not in {"success", ""}):
                raise LocalH3Error("ComfyUI execution failed: " + _status_error(entry))
            if status.get("completed") and status_str == "success":
                videos = _saved_videos(entry)
                if not videos:
                    raise LocalH3Error("ComfyUI completed without an mp4 in history outputs")
                return videos[-1]
        sleep(poll)
    client.cancel(prompt_id)
    suffix = f" Last error: {last_error}" if last_error else ""
    raise LocalH3Error(f"timed out after {int(timeout)}s waiting for local H3 prompt {prompt_id}.{suffix}")


def _destination(output_dir: str, output_path: Optional[str]) -> Path:
    if resolve_video_output_dir is None or default_video_output_dir is None:
        raise LocalH3Error("capsule output guard is unavailable")
    from src.utils.output_paths import resolve_project_path

    default_dir = default_video_output_dir("local_comfy_minimax_h3")
    if output_path:
        destination = resolve_project_path(output_path)
        checked = Path(resolve_video_output_dir(str(destination.parent), str(destination), default_dir)).resolve(strict=False)
        if checked != destination.parent.resolve(strict=False):
            raise LocalH3Error(f"output path is outside the capsule output directory: {destination}")
        return destination
    directory = Path(resolve_video_output_dir(output_dir or None, None, default_dir))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"local_comfy_h3_{uuid.uuid4().hex[:8]}.mp4"


def _write_mp4(destination: Path, payload: bytes) -> None:
    if not is_valid_mp4(payload):
        raise LocalH3Error(
            f"ComfyUI returned {len(payload)} bytes that are not a valid mp4; refusing to save them"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def uploads_for_plan(client: ComfyClient, plan: Mapping[str, Any]) -> dict[str, Any]:
    token = uuid.uuid4().hex[:8]
    if plan["mode"] == "reference_to_video":
        return {"refs": [client.upload_image(path, f"h3_ref_{token}_{index}") for index, path in enumerate(plan["refs"], start=1)]}
    uploaded: dict[str, Any] = {}
    if plan["first"]:
        uploaded["first"] = client.upload_image(plan["first"], f"h3_first_{token}")
    if plan["last"]:
        uploaded["last"] = client.upload_image(plan["last"], f"h3_last_{token}")
    return uploaded


def execute_plan(
    plan: Mapping[str, Any],
    *,
    output_dir: str,
    output_path: Optional[str],
    comfy_url: str,
    dry_run: bool,
    transport: Optional[UrllibTransport] = None,
    client: Optional[ComfyClient] = None,
) -> dict[str, Any]:
    uploaded = placeholder_uploads(plan)
    graph = graph_from_plan(plan, uploaded)
    public_plan = {key: value for key, value in plan.items() if key != "prompt"}
    public_plan["prompt_characters"] = len(plan["prompt"])
    if dry_run:
        return {
            "success": True,
            "dry_run": True,
            "network": False,
            "engine": ENGINE,
            "provider": PROVIDER,
            "plan": public_plan,
            "graph_class_types": sorted({node["class_type"] for node in graph.values()}),
            "graph": graph,
        }
    destination = _destination(output_dir, output_path)
    active = client or ComfyClient(comfy_url, transport=transport)
    names = list(FL2VA_NODES)
    if plan["mode"] == "reference_to_video":
        names = [name for name in names if name != "MiniMaxH3ImageToVideo"] + ["MiniMaxH3ReferenceToVideo"]
    if not plan.get("lora"):
        names = [name for name in names if name != "LoraLoaderModelOnly"]
    info = active.object_info(names)
    preflight_object_info(info, plan)
    real_uploads = uploads_for_plan(active, plan)
    graph = graph_from_plan(plan, real_uploads)
    client_id = str(uuid.uuid4())
    prompt_id = active.submit(graph, client_id)
    saved = poll_until_video(active, prompt_id, timeout_seconds(plan["profile"]), poll_seconds())
    payload = active.download(saved["filename"], saved["subfolder"], saved["type"])
    _write_mp4(destination, payload)
    return {
        "success": True,
        "dry_run": False,
        "engine": ENGINE,
        "provider": PROVIDER,
        "output_path": str(destination),
        "video_path": str(destination),
        "prompt_id": prompt_id,
        "comfy_file": saved,
        "width": plan["width"],
        "height": plan["height"],
        "length": plan["length"],
        "steps": plan["steps"],
        "seed": plan["seed"],
        "profile": plan["profile"],
        "mode": plan["mode"],
        "unet": plan["unet"],
        "lora": plan.get("lora"),
        "sampler": plan["sampler"],
    }


def generation_request_from_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return dict(values)


def rewrite_jobs(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Point capsule_cinema jobs at the local engine without changing prompts or frames."""
    rewritten = deepcopy(dict(spec))
    jobs = rewritten.get("jobs")
    if not isinstance(jobs, list):
        raise LocalH3Error("jobs.json has no jobs list")
    changed = 0
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        if job.get("executor") != "capsule_cinema":
            continue
        params = job.get("params")
        if not isinstance(params, dict):
            raise LocalH3Error(f"shot {job.get('shot')} is missing params")
        params["engine"] = ENGINE
        changed += 1
    if changed == 0:
        raise LocalH3Error("jobs.json has no capsule_cinema generation jobs to retarget")
    rewritten["engine"] = ENGINE
    rewritten["local_h3_channel"] = ENGINE
    return rewritten


class LocalComfyMiniMaxH3VideoGeneratorTool(BaseTool):
    name: str = "Local ComfyUI MiniMax H3 video generator"
    description: str = (
        "Generate video on the local loopback ComfyUI MiniMax H3 FL2VA or Ref2VA graph. "
        "Does not call paid MiniMax, Hailuo, RunningHub, or Comfy Cloud APIs."
    )
    args_schema: type[BaseModel] = LocalComfyMiniMaxH3Schema

    def _run(self, prompt: str, output_dir: str = "", engine: str = ENGINE, **kwargs: Any) -> dict[str, Any]:
        request = {"prompt": prompt, "output_dir": output_dir, "engine": engine, **kwargs}
        try:
            plan = plan_generation(request)
            return execute_plan(
                plan,
                output_dir=output_dir,
                output_path=kwargs.get("output_path"),
                comfy_url=str(kwargs.get("comfy_url") or comfy_url_from_env()),
                dry_run=bool(kwargs.get("dry_run")),
            )
        except LocalH3Error as exc:
            return {
                "success": False,
                "engine": ENGINE,
                "provider": PROVIDER,
                "error": str(exc),
            }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _command_dry_run(args: argparse.Namespace) -> int:
    if args.jobs:
        spec = _load_json(Path(args.jobs))
        jobs = [job for job in spec.get("jobs") or [] if job.get("executor") == "capsule_cinema"]
        if args.shot is not None:
            jobs = [job for job in jobs if int(job.get("shot")) == int(args.shot)]
        if not jobs:
            raise LocalH3Error("no capsule_cinema job matched")
        request = dict(jobs[0]["params"])
        current_engine = str(request.get("engine") or "")
        if current_engine and current_engine != ENGINE:
            raise LocalH3Error(
                f"job engine is {current_engine!r}; run rewrite-jobs to select {ENGINE} explicitly. "
                "This dry-run will not retarget a paid engine."
            )
    else:
        request = {
            "prompt": args.prompt,
            "generation_type": args.generation_type,
            "engine": ENGINE,
            "aspect_ratio": args.aspect_ratio,
            "duration": args.duration,
            "start_image_path": args.first,
            "end_image_path": args.last,
            "image_path": args.image,
            "image_paths": args.image_path or [],
            "profile": args.profile,
            "seed": args.seed,
        }
    request["engine"] = request.get("engine") or ENGINE
    if args.profile:
        request["profile"] = args.profile
    plan = plan_generation(request)
    result = execute_plan(plan, output_dir="", output_path=None, comfy_url=comfy_url_from_env(), dry_run=True)
    result["graph"] = {key: value["class_type"] for key, value in result["graph"].items()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _command_rewrite(args: argparse.Namespace) -> int:
    spec = rewrite_jobs(_load_json(Path(args.jobs)))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(destination)
    return 0


def _command_generate(args: argparse.Namespace) -> int:
    request = {
        "prompt": args.prompt,
        "generation_type": args.generation_type,
        "engine": ENGINE,
        "output_dir": str(Path(args.output).expanduser().parent),
        "output_path": args.output,
        "aspect_ratio": args.aspect_ratio,
        "duration": args.duration,
        "start_image_path": args.first,
        "end_image_path": args.last,
        "image_path": args.image,
        "image_paths": args.image_path or [],
        "profile": args.profile,
        "seed": args.seed,
        "comfy_url": args.comfy_url,
    }
    tool = LocalComfyMiniMaxH3VideoGeneratorTool()
    result = tool._run(**request)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("success") and result.get("output_path") else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local loopback ComfyUI MiniMax H3 channel")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser, *, with_output: bool) -> None:
        target.add_argument("--prompt", default="")
        target.add_argument("--generation-type", default="first_last_frame")
        target.add_argument("--aspect-ratio", default="9:16")
        target.add_argument("--duration", type=float, default=4)
        target.add_argument("--first", default="")
        target.add_argument("--last", default="")
        target.add_argument("--image", default="")
        target.add_argument("--image-path", action="append")
        target.add_argument("--profile", default="")
        target.add_argument("--seed", type=int)
        target.add_argument("--jobs", default="")
        target.add_argument("--shot", type=int)
        if with_output:
            target.add_argument("--output", required=True)
            target.add_argument("--comfy-url", default="")

    dry = sub.add_parser("dry-run", help="Print the local graph. Does not open a network connection.")
    add_common(dry, with_output=False)
    dry.set_defaults(func=_command_dry_run)

    rewrite = sub.add_parser("rewrite-jobs", help="Write a jobs.json whose capsule engine is the local H3 channel")
    rewrite.add_argument("--jobs", required=True)
    rewrite.add_argument("--output", required=True)
    rewrite.set_defaults(func=_command_rewrite)

    generate = sub.add_parser("generate", help="Submit a real local ComfyUI job and save the mp4")
    add_common(generate, with_output=True)
    generate.set_defaults(func=_command_generate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except LocalH3Error as exc:
        print(json.dumps({"success": False, "engine": ENGINE, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
