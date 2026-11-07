"""Copy the local H3 channel into a capsule-cinema checkout.

The adapter filename matches the upgrade preserve rule
``lib/custom_tools/**/local_*_adapter*.py``. Registry records go in the
Git-ignored ``local-channels`` directory, which upgrades also preserve.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml


TOOL_NAME = "LocalComfyMiniMaxH3VideoGeneratorTool"
ADAPTER_NAME = "local_comfy_minimax_h3_adapter.py"


def _merge_yaml(source: Path, destination: Path, mapping_key: str) -> None:
    incoming = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    current = {}
    if destination.is_file():
        current = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    if not isinstance(current, dict):
        current = {}
    if not isinstance(incoming, dict):
        raise SystemExit(f"{source} is not a mapping")
    merged = dict(current)
    for key, value in incoming.items():
        if key == mapping_key and isinstance(value, dict):
            existing = merged.get(mapping_key)
            if not isinstance(existing, dict):
                existing = {}
            existing.update(value)
            merged[mapping_key] = existing
        elif key not in merged:
            merged[key] = value
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(merged, sort_keys=False, allow_unicode=True), encoding="utf-8")


def install(capsule_root: Path) -> list[Path]:
    root = capsule_root.expanduser().resolve()
    package = Path(__file__).resolve().parent
    if not (root / "lib" / "custom_tools").is_dir():
        raise SystemExit(f"not a capsule-cinema checkout: {root}")
    adapter_destination = root / "lib" / "custom_tools" / "video_generation" / ADAPTER_NAME
    adapter_destination.parent.mkdir(parents=True, exist_ok=True)
    source_adapter = package / ADAPTER_NAME
    if source_adapter.resolve() != adapter_destination.resolve():
        shutil.copy2(source_adapter, adapter_destination)
    channels = root / "local-channels"
    _merge_yaml(package / "registry" / "tool_registry.yaml", channels / "tool_registry.yaml", "tools")
    _merge_yaml(
        package / "registry" / "tool_capabilities.yaml",
        channels / "tool_capabilities.yaml",
        "tools",
    )
    written = [
        adapter_destination,
        channels / "tool_registry.yaml",
        channels / "tool_capabilities.yaml",
    ]
    registry = yaml.safe_load((channels / "tool_registry.yaml").read_text(encoding="utf-8"))
    record = (registry.get("tools") or {}).get(TOOL_NAME) or {}
    if record.get("runtime_engine") != "local-comfy-minimax-h3":
        raise SystemExit("registry merge did not keep runtime_engine local-comfy-minimax-h3")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule-root", required=True)
    args = parser.parse_args(argv)
    written = install(Path(args.capsule_root))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
