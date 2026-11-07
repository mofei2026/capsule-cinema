"""Offline tests for the local ComfyUI MiniMax H3 channel."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
LIB = ROOT / "lib"
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(ROOT / "scripts"))

os.environ.pop("CAPSULE_LOCAL_H3_PROFILE", None)
os.environ.pop("CAPSULE_LOCAL_H3_SEED", None)
os.environ.pop("CAPSULE_LOCAL_H3_TIMEOUT_SECONDS", None)
os.environ["CAPSULE_LOCAL_COMFY_URL"] = "http://127.0.0.1:8188"


def _load_adapter():
    path = PACKAGE / "local_comfy_minimax_h3_adapter.py"
    spec = importlib.util.spec_from_file_location("local_comfy_minimax_h3_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


h3 = _load_adapter()


def _mp4() -> bytes:
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8 + b"\x00" * 5000


def _object_info(module, *, drop=(), unet_names=None) -> dict:
    unets = unet_names or [module.FL2VA_UNET, module.REF2VA_UNET]
    present = {
        "UNETLoader": {"input": {"required": {"unet_name": [unets, {}], "weight_dtype": [["default"], {}]}}},
        "CLIPLoader": {
            "input": {
                "required": {
                    "clip_name": [[module.TEXT_ENCODER], {}],
                    "type": [["minimax", "stable_diffusion"], {}],
                }
            }
        },
        "VAELoader": {"input": {"required": {"vae_name": [[module.VIDEO_VAE, module.AUDIO_VAE], {}]}}},
        "LoraLoaderModelOnly": {
            "input": {"required": {"lora_name": [[module.FL2VA_LORA, module.REF2VA_LORA], {}]}}
        },
        "KSamplerSelect": {"input": {"required": {"sampler_name": ["COMBO", {"options": ["euler", "res_multistep"]}]}}},
        "BasicScheduler": {"input": {"required": {"scheduler": ["COMBO", {"options": ["simple", "sgm_uniform"]}]}}},
        "LoadImage": {"input": {"required": {"image": [["example.png"], {}]}}},
        "MiniMaxH3ImageToVideo": {"input": {"required": {}}},
        "MiniMaxH3ReferenceToVideo": {"input": {"required": {}}},
        "BasicGuider": {"input": {"required": {}}},
        "RandomNoise": {"input": {"required": {}}},
        "SamplerCustomAdvanced": {"input": {"required": {}}},
        "VAEDecode": {"input": {"required": {}}},
        "VAEDecodeAudio": {"input": {"required": {}}},
        "CreateVideo": {"input": {"required": {}}},
        "SaveVideo": {"input": {"required": {}}},
    }
    for name in drop:
        present.pop(name, None)
    return present


class ScriptedTransport:
    def __init__(self, info: dict, *, history_script=None, view=None, prompt_error=None):
        self.info = info
        self.calls = []
        self.history_script = list(history_script or [])
        self.view = view if view is not None else _mp4()
        self.prompt_error = prompt_error
        self.prompt_id = "11111111-1111-1111-1111-111111111111"

    def request(self, method, url, data=None, headers=None, timeout=None):
        parsed = urlparse(url)
        self.calls.append({"method": method, "path": parsed.path, "query": parse_qs(parsed.query), "data": data, "headers": headers})
        if parsed.path.startswith("/object_info/"):
            name = parsed.path.split("/", 2)[-1]
            payload = {name: self.info[name]} if name in self.info else {}
            return 200, json.dumps(payload).encode()
        if parsed.path == "/upload/image":
            body = data or b""
            if b'name="image"' not in body or b"dry-image" not in body and b"PNG" not in body and len(body) < 20:
                return 400, b'{"error":"empty upload"}'
            return 200, json.dumps({"name": "uploaded.png", "subfolder": "local_h3", "type": "input"}).encode()
        if parsed.path == "/prompt" and method == "POST":
            posted = json.loads(data.decode())
            if "client_id" not in posted or "prompt" not in posted:
                return 400, b'{"error":"missing prompt"}'
            if self.prompt_error:
                return 400, json.dumps({"error": self.prompt_error, "node_errors": {"8": {"errors": [{"message": "bad"}]}}}).encode()
            return 200, json.dumps({"prompt_id": self.prompt_id, "number": 1, "node_errors": {}}).encode()
        if parsed.path == "/queue" and method == "GET":
            return 200, json.dumps({"queue_running": [[0, self.prompt_id]], "queue_pending": []}).encode()
        if parsed.path == "/queue" and method == "POST":
            return 200, b"{}"
        if parsed.path == "/interrupt":
            return 200, b""
        if parsed.path.startswith("/history/"):
            if self.history_script:
                entry = self.history_script.pop(0)
            else:
                entry = {
                    "status": {"status_str": "success", "completed": True, "messages": []},
                    "outputs": {"17": {"images": [{"filename": "local_h3_00001_.mp4", "subfolder": "local_h3", "type": "output"}], "animated": [True]}},
                }
            if entry is None:
                return 200, b"{}"
            return 200, json.dumps({self.prompt_id: entry}).encode()
        if parsed.path == "/view":
            return 200, self.view
        return 404, b'{"error":"not mocked"}'


class PlanningTests(unittest.TestCase):
    def test_rea_first_last_job_maps_to_fl2va_not_ref(self):
        request = {
            "prompt": "主体从首帧走到尾帧",
            "output_dir": r"D:\Apps\capsule-cinema\output\rea_remake\shot_001",
            "generation_type": "first_last_frame",
            "engine": "local-comfy-minimax-h3",
            "aspect_ratio": "9:16",
            "duration": 4,
            "start_image_path": r"D:\Apps\rea-video\verify-out\synth_ai\frames\shot_001_first.png",
            "end_image_path": r"D:\Apps\rea-video\verify-out\synth_ai\frames\shot_001_last.png",
        }
        plan = h3.plan_generation(request)
        graph = h3.graph_from_plan(plan, h3.placeholder_uploads(plan))
        self.assertEqual(plan["mode"], "first_last_frame")
        self.assertEqual(plan["family"], "fl2va")
        self.assertEqual(plan["unet"], h3.FL2VA_UNET)
        self.assertEqual(plan["width"], 768)
        self.assertEqual(plan["height"], 1344)
        self.assertEqual(plan["length"], 107)
        self.assertEqual(plan["steps"], 8)
        self.assertEqual(plan["sampler"], "res_multistep")
        self.assertEqual(plan["lora"], h3.FL2VA_LORA)
        self.assertEqual(plan["seed"], 5555)
        self.assertNotIn("MiniMaxH3ReferenceToVideo", {node["class_type"] for node in graph.values()})
        image_node = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertIn("first_frame", image_node["inputs"])
        self.assertIn("last_frame", image_node["inputs"])
        save = next(node for node in graph.values() if node["class_type"] == "SaveVideo")
        self.assertEqual(save["inputs"]["format"], "mp4")
        self.assertEqual(save["inputs"]["format.codec"], "h264")
        self.assertNotIn("CFGGuider", {node["class_type"] for node in graph.values()})

    def test_smoke_profile_is_small_and_deterministic(self):
        plan = h3.plan_generation(
            {
                "prompt": "smoke",
                "engine": h3.ENGINE,
                "generation_type": "first_last_frame",
                "aspect_ratio": "9:16",
                "duration": 4,
                "start_image_path": "a.png",
                "end_image_path": "b.png",
                "profile": "smoke",
            }
        )
        self.assertEqual((plan["width"], plan["height"], plan["length"], plan["steps"], plan["seed"]), (384, 672, 22, 8, 5555))
        self.assertEqual(plan["lora"], h3.FL2VA_LORA)

    def test_quality_profile_keeps_official_canvas_without_turbo_lora(self):
        plan = h3.plan_generation(
            {
                "prompt": "quality",
                "engine": h3.ENGINE,
                "generation_type": "first_last_frame",
                "aspect_ratio": "16:9",
                "duration": 5,
                "start_image_path": "a.png",
                "end_image_path": "b.png",
                "profile": "quality",
            }
        )
        self.assertEqual((plan["width"], plan["height"]), (1344, 768))
        self.assertEqual(plan["length"], 124)
        self.assertEqual(plan["steps"], 20)
        self.assertIsNone(plan["lora"])

    def test_reference_mode_is_not_used_for_first_last(self):
        with self.assertRaises(h3.LocalH3Error):
            h3.plan_generation(
                {
                    "prompt": "refs",
                    "engine": h3.ENGINE,
                    "generation_type": "first_last_frame",
                    "start_image_path": "a.png",
                    "end_image_path": "b.png",
                    "image_paths": ["r1.png", "r2.png", "r3.png"],
                }
            )
        plan = h3.plan_generation(
            {
                "prompt": "refs",
                "engine": h3.ENGINE,
                "generation_type": "reference_to_video",
                "aspect_ratio": "16:9",
                "duration": 4,
                "image_paths": [r"D:\refs\1.png", r"D:\refs\2.png"],
                "profile": "turbo",
            }
        )
        graph = h3.graph_from_plan(plan, h3.placeholder_uploads(plan))
        classes = {node["class_type"] for node in graph.values()}
        self.assertIn("MiniMaxH3ReferenceToVideo", classes)
        self.assertNotIn("MiniMaxH3ImageToVideo", classes)
        self.assertEqual(plan["unet"], h3.REF2VA_UNET)
        self.assertEqual(plan["sampler"], "euler")
        self.assertEqual(plan["steps"], 4)
        self.assertEqual(plan["lora"], h3.REF2VA_LORA)
        ref = next(node for node in graph.values() if node["class_type"] == "MiniMaxH3ReferenceToVideo")
        self.assertIn("ref_images.ref_image_1", ref["inputs"])
        self.assertIn("ref_images.ref_image_2", ref["inputs"])

    def test_paid_engines_fail_closed(self):
        for engine in ("minimax-h3", "seedance2.0", "hailuo-03"):
            with self.assertRaises(h3.LocalH3Error):
                h3.plan_generation(
                    {
                        "prompt": "x",
                        "engine": engine,
                        "generation_type": "text_to_video",
                    }
                )

    def test_unsafe_endpoints_and_music_port(self):
        for url in (
            "http://192.168.1.20:8188",
            "https://api.minimaxi.com",
            "http://127.0.0.1:7860",
            "http://example.com:8188",
        ):
            with self.assertRaises(h3.LocalH3Error):
                h3.assert_loopback_url(url)
        self.assertEqual(h3.assert_loopback_url("http://127.0.0.1:8188"), "http://127.0.0.1:8188")

    def test_missing_frames_and_unsupported_mode(self):
        with self.assertRaises(h3.LocalH3Error):
            h3.resolve_mode_images({"generation_type": "first_last_frame", "start_image_path": "a.png"})
        with self.assertRaises(h3.LocalH3Error):
            h3.resolve_mode_images({"generation_type": "image_to_video", "image_path": "a.png", "end_image_path": "b.png"})
        with self.assertRaises(h3.LocalH3Error):
            h3.resolve_mode_images({"generation_type": "lip_sync"})

    def test_rewrite_jobs_switches_engine_without_touching_prompt(self):
        spec = {
            "capsule_root": r"D:\Apps\capsule-cinema",
            "engine": "seedance2.0",
            "mode": "first_last_frame",
            "jobs": [
                {
                    "shot": 1,
                    "executor": "capsule_cinema",
                    "params": {
                        "prompt": "保持主体",
                        "generation_type": "first_last_frame",
                        "engine": "seedance2.0",
                        "aspect_ratio": "9:16",
                        "duration": 4,
                        "start_image_path": r"D:\frames\first.png",
                        "end_image_path": r"D:\frames\last.png",
                        "output_dir": r"D:\Apps\capsule-cinema\output\rea\shot_001",
                    },
                },
                {"shot": 2, "executor": "keep_original"},
            ],
        }
        rewritten = h3.rewrite_jobs(spec)
        self.assertEqual(spec["jobs"][0]["params"]["engine"], "seedance2.0")
        self.assertEqual(rewritten["engine"], h3.ENGINE)
        params = rewritten["jobs"][0]["params"]
        self.assertEqual(params["engine"], h3.ENGINE)
        self.assertEqual(params["prompt"], "保持主体")
        self.assertEqual(params["generation_type"], "first_last_frame")
        self.assertEqual(params["start_image_path"], r"D:\frames\first.png")
        self.assertEqual(params["duration"], 4)
        self.assertEqual(rewritten["jobs"][1]["executor"], "keep_original")

    def test_preflight_rejects_missing_node_or_model(self):
        plan = h3.plan_generation(
            {
                "prompt": "x",
                "engine": h3.ENGINE,
                "generation_type": "text_to_video",
                "profile": "turbo",
            }
        )
        with self.assertRaises(h3.LocalH3Error):
            h3.preflight_object_info(_object_info(h3, drop=("MiniMaxH3ImageToVideo",)), plan)
        with self.assertRaises(h3.LocalH3Error):
            h3.preflight_object_info(_object_info(h3, unet_names=["other.safetensors"]), plan)

    def test_dry_run_does_not_touch_transport(self):
        plan = h3.plan_generation(
            {
                "prompt": "x",
                "engine": h3.ENGINE,
                "generation_type": "text_to_video",
                "profile": "smoke",
            }
        )

        class Boom:
            def request(self, *args, **kwargs):
                raise AssertionError("dry-run opened the network")

        result = h3.execute_plan(
            plan,
            output_dir="",
            output_path=None,
            comfy_url="http://127.0.0.1:8188",
            dry_run=True,
            transport=Boom(),
        )
        self.assertTrue(result["dry_run"])
        self.assertFalse(result["network"])
        self.assertEqual(result["engine"], h3.ENGINE)
        self.assertIn("MiniMaxH3ImageToVideo", result["graph_class_types"])
        self.assertNotIn("MiniMaxH3ReferenceToVideo", result["graph_class_types"])


class EndpointTests(unittest.TestCase):
    def _plan(self):
        return h3.plan_generation(
            {
                "prompt": "a short local move",
                "engine": h3.ENGINE,
                "generation_type": "first_last_frame",
                "aspect_ratio": "9:16",
                "duration": 4,
                "start_image_path": "first.png",
                "end_image_path": "last.png",
                "profile": "smoke",
                "seed": 5555,
            }
        )

    def test_success_downloads_valid_mp4(self):
        info = _object_info(h3)
        transport = ScriptedTransport(info)
        client = h3.ComfyClient("http://127.0.0.1:8188", transport=transport)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            last = Path(directory) / "last.png"
            first.write_bytes(b"\x89PNG\r\n\x1a\n" + b"dry-image")
            last.write_bytes(b"\x89PNG\r\n\x1a\n" + b"dry-image")
            plan = self._plan()
            plan["first"] = str(first)
            plan["last"] = str(last)
            destination = ROOT / "output" / "local_h3_tests" / "success.mp4"
            if destination.exists():
                destination.unlink()
            result = h3.execute_plan(
                plan,
                output_dir=str(destination.parent),
                output_path=str(destination),
                comfy_url="http://127.0.0.1:8188",
                dry_run=False,
                client=client,
            )
        self.assertTrue(result["success"])
        self.assertTrue(destination.is_file())
        self.assertTrue(h3.is_valid_mp4(destination.read_bytes()))
        prompt_calls = [call for call in transport.calls if call["path"] == "/prompt"]
        posted = json.loads(prompt_calls[0]["data"].decode())
        classes = {node["class_type"] for node in posted["prompt"].values()}
        self.assertIn("MiniMaxH3ImageToVideo", classes)
        self.assertNotIn("MiniMaxH3ReferenceToVideo", classes)
        self.assertTrue(posted["client_id"])
        image_node = next(node for node in posted["prompt"].values() if node["class_type"] == "MiniMaxH3ImageToVideo")
        self.assertEqual(image_node["inputs"]["width"], 384)
        self.assertEqual(image_node["inputs"]["height"], 672)
        self.assertEqual(image_node["inputs"]["length"], 22)
        self.assertTrue(any(call["path"] == "/queue" and call["method"] == "GET" for call in transport.calls))
        self.assertTrue(any(call["path"].startswith("/history/") for call in transport.calls))
        self.assertTrue(any(call["path"] == "/view" for call in transport.calls))

    def test_execution_error_and_invalid_video_and_timeout(self):
        info = _object_info(h3)
        error_transport = ScriptedTransport(
            info,
            history_script=[
                {
                    "status": {
                        "status_str": "error",
                        "completed": True,
                        "messages": [["execution_error", {"exception_type": "RuntimeError", "exception_message": "oom", "node_type": "UNETLoader"}]],
                    },
                    "outputs": {},
                }
            ],
        )
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            last = Path(directory) / "last.png"
            first.write_bytes(b"\x89PNG\r\n\x1a\n" + b"dry-image")
            last.write_bytes(b"\x89PNG\r\n\x1a\n" + b"dry-image")
            plan["first"] = str(first)
            plan["last"] = str(last)
            with self.assertRaises(h3.LocalH3Error) as caught:
                h3.execute_plan(
                    plan,
                    output_dir=str(ROOT / "output" / "local_h3_tests"),
                    output_path=str(ROOT / "output" / "local_h3_tests" / "error.mp4"),
                    comfy_url="http://127.0.0.1:8188",
                    dry_run=False,
                    client=h3.ComfyClient("http://127.0.0.1:8188", transport=error_transport),
                )
            self.assertIn("oom", str(caught.exception))

            bad = ScriptedTransport(info, view=b"<html>" + b"x" * 5000)
            with self.assertRaises(h3.LocalH3Error) as caught:
                h3.execute_plan(
                    plan,
                    output_dir=str(ROOT / "output" / "local_h3_tests"),
                    output_path=str(ROOT / "output" / "local_h3_tests" / "bad.mp4"),
                    comfy_url="http://127.0.0.1:8188",
                    dry_run=False,
                    client=h3.ComfyClient("http://127.0.0.1:8188", transport=bad),
                )
            self.assertIn("not a valid mp4", str(caught.exception))
            self.assertFalse((ROOT / "output" / "local_h3_tests" / "bad.mp4").exists())

        class Never:
            def __init__(self):
                self.cancelled = []
                self.now = 0

            def queue(self):
                return {"queue_running": [[1, "prompt-1"]], "queue_pending": []}

            def history(self, prompt_id):
                return None

            def cancel(self, prompt_id):
                self.cancelled.append(prompt_id)

        clock_client = Never()

        def clock():
            clock_client.now += 5
            return clock_client.now

        with self.assertRaises(h3.LocalH3Error) as caught:
            h3.poll_until_video(clock_client, "prompt-1", timeout=12, poll=1, clock=clock, sleep=lambda _seconds: None)
        self.assertIn("timed out", str(caught.exception))
        self.assertEqual(clock_client.cancelled, ["prompt-1"])

    def test_cancel_targets_only_our_prompt(self):
        transport = ScriptedTransport(_object_info(h3))
        client = h3.ComfyClient("http://127.0.0.1:8188", transport=transport)
        client.cancel("prompt-1")
        posted = [json.loads(call["data"].decode()) for call in transport.calls if call["method"] == "POST"]
        self.assertIn({"delete": ["prompt-1"]}, posted)
        self.assertIn({"prompt_id": "prompt-1"}, posted)
        self.assertFalse(any("clear" in item for item in posted))

    def test_prompt_node_errors_fail_closed(self):
        transport = ScriptedTransport(_object_info(h3), prompt_error={"type": "prompt_outputs_failed_validation"})
        client = h3.ComfyClient("http://127.0.0.1:8188", transport=transport)
        with self.assertRaises(h3.LocalH3Error):
            client.submit({"1": {"class_type": "SaveVideo", "inputs": {}}}, "client")

    def test_registry_selects_local_engine_not_paid_minimax(self):
        with tempfile.TemporaryDirectory() as directory:
            channels = Path(directory)
            (channels / "tool_registry.yaml").write_text(
                (PACKAGE / "registry" / "tool_registry.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            os.environ["CAPSULE_CINEMA_LOCAL_CHANNELS_DIR"] = str(channels)
            try:
                module_name = "custom_tools.video_generation.local_comfy_minimax_h3_adapter"
                sys.modules[module_name] = h3
                from importlib import import_module

                from src.config_registry import load_tool_registry

                def tool_for(engine: str):
                    records = load_tool_registry()
                    for class_name, record in records.items():
                        if not isinstance(record, dict) or record.get("category") != "video_generation":
                            continue
                        if record.get("runtime_engine") == engine:
                            return getattr(import_module(record["module"]), class_name)()
                    raise AssertionError(engine)

                tool = tool_for(h3.ENGINE)
                self.assertIsInstance(tool, h3.LocalComfyMiniMaxH3VideoGeneratorTool)
                paid = tool_for("minimax-h3")
                self.assertNotIsInstance(paid, h3.LocalComfyMiniMaxH3VideoGeneratorTool)
                self.assertEqual(paid.__class__.__name__, "MiniMaxH3VideoGeneratorTool")
            finally:
                os.environ.pop("CAPSULE_CINEMA_LOCAL_CHANNELS_DIR", None)
                sys.modules.pop("custom_tools.video_generation.local_comfy_minimax_h3_adapter", None)


class InstallTests(unittest.TestCase):
    def test_install_merges_registry_and_preserves_other_tools(self):
        sys.path.insert(0, str(PACKAGE))
        import install_local_channel

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib" / "custom_tools" / "video_generation").mkdir(parents=True)
            channels = root / "local-channels"
            channels.mkdir()
            (channels / "tool_registry.yaml").write_text(
                "version: 2\ntools:\n  ExistingTool:\n    module: custom_tools.example\n    category: video_generation\n    runtime_engine: example\n",
                encoding="utf-8",
            )
            written = install_local_channel.install(root)
            self.assertTrue((root / "lib" / "custom_tools" / "video_generation" / "local_comfy_minimax_h3_adapter.py").is_file())
            import yaml

            registry = yaml.safe_load((channels / "tool_registry.yaml").read_text(encoding="utf-8"))
            self.assertIn("ExistingTool", registry["tools"])
            record = registry["tools"]["LocalComfyMiniMaxH3VideoGeneratorTool"]
            self.assertEqual(record["runtime_engine"], "local-comfy-minimax-h3")
            self.assertTrue(written)


if __name__ == "__main__":
    unittest.main()
