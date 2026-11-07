# FM 本地 MiniMax H3 通道

这个目录是 capsule-cinema 的本地扩展。它只向本机回环上的 ComfyUI 提交图，不调用 MiniMax、Hailuo、RunningHub 或 Comfy Cloud，也不下载权重。

rea-video `remake-run` 会把 `jobs.json` 里的 `params.engine` 原样交给 `UniversalVideoGenerationTool`。公开引擎名 `minimax-h3` 仍然是官方付费接口。本地通道的引擎名是：

```text
local-comfy-minimax-h3
```

不传这个名字就不会进入本通道。失败时不会改走 Seedance 或官方 H3。

## 安装后的文件

在 `D:\Apps\capsule-cinema` 执行安装脚本后，文件落点是：

| 源文件 | 安装位置 | 升级时 |
| --- | --- | --- |
| `local_comfy_minimax_h3_adapter.py` | `lib\custom_tools\video_generation\local_comfy_minimax_h3_adapter.py` | 安装器会保留 `local_*_adapter*.py` |
| `registry\tool_registry.yaml` | `local-channels\tool_registry.yaml`（只合并本工具） | `local-channels` 在升级时保留 |
| `registry\tool_capabilities.yaml` | `local-channels\tool_capabilities.yaml`（只合并本工具） | 同上 |

不要改 ComfyUI 根目录，不要动 `E:\ComfyUI_SoulX` 里的模型，也不要向 `7860` 端口提交任务。那个端口是 ACE-Step。

可选的 rea-video 补丁 `rea-video-windows-python.patch` 只改解释器查找：在未设置 `CAPSULE_CINEMA_PYTHON` 时，同时接受 `.venv\Scripts\python.exe`。FM 已经设置了该变量，补丁不是生成视频的前提。

## 依赖

capsule-cinema 现有的 Python 3.12 虚拟环境即可（`pydantic`、`PyYAML` 已在 `lib/requirements.txt`）。不要新增云端 SDK，也不要给 ComfyUI 装节点或下权重。

ComfyUI 必须已经在 `http://127.0.0.1:8188` 运行，并且 `object_info` 里能看到这些已安装文件：

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`

显式 `reference_to_video` 才会用 Ref2VA int8 和 4-step Ref2VA LoRA。`first_last_frame` 不会走 Ref2VA。

## 持久配置

写在用户环境或 capsule 的 `.env`，不要写入仓库：

```text
CAPSULE_LOCAL_COMFY_URL=http://127.0.0.1:8188
CAPSULE_LOCAL_H3_PROFILE=turbo
CAPSULE_LOCAL_H3_SEED=5555
CAPSULE_LOCAL_H3_TIMEOUT_SECONDS=3600
CAPSULE_LOCAL_H3_POLL_SECONDS=3
```

`PROFILE`：

- `smoke`：9:16 时为 384×672、22 帧、8 步、FL2VA turbo LoRA、种子 5555。只用于接线冒烟。
- `turbo`：按官方画布（9:16 为 768×1344）和时长对齐帧数，8 步，FL2VA turbo LoRA。这是正式复刻的默认。
- `quality`：同一画布，20 步，不加载 turbo LoRA。

4 秒会落到模型的 17k+5 网格，即 107 帧（24 fps）。5 秒是 124 帧。冒烟固定 22 帧，短于训练区间，只证明链路，不代表成片质量。

非回环地址、带账号的 URL，以及端口 7860，会直接拒绝。

## 安装

在 FM 上，把本目录放到任意路径，例如 `D:\Apps\fm-local-comfy-h3`，然后：

```bat
D:\Apps\capsule-cinema\.venv\Scripts\python.exe D:\Apps\fm-local-comfy-h3\install_local_channel.py --capsule-root D:\Apps\capsule-cinema
```

确认输出里有适配器路径，以及 `local-channels\tool_registry.yaml`。

可选 rea 补丁（在 `D:\Apps\rea-video`）：

```bat
git apply --check D:\Apps\fm-local-comfy-h3\rea-video-windows-python.patch
git apply D:\Apps\fm-local-comfy-h3\rea-video-windows-python.patch
```

## 不计费 dry-run

下面这条不连接 ComfyUI，也不调用付费接口。它直接构建 FL2VA 图：

```bat
D:\Apps\capsule-cinema\.venv\Scripts\python.exe D:\Apps\capsule-cinema\lib\custom_tools\video_generation\local_comfy_minimax_h3_adapter.py dry-run --profile smoke --generation-type first_last_frame --aspect-ratio 9:16 --duration 4 --prompt "主体从首帧自然过渡到尾帧" --first D:\frames\first.png --last D:\frames\last.png
```

输出里的 `engine` 必须是 `local-comfy-minimax-h3`，`network` 为 false，节点里有 `MiniMaxH3ImageToVideo`，不能出现 `MiniMaxH3ReferenceToVideo`。

把已有的 seedance 任务改成显式本地引擎（不改提示词和首尾帧，写到新文件）：

```bat
D:\Apps\capsule-cinema\.venv\Scripts\python.exe D:\Apps\capsule-cinema\lib\custom_tools\video_generation\local_comfy_minimax_h3_adapter.py rewrite-jobs --jobs D:\Apps\rea-video\verify-out\synth_ai\remake\jobs.json --output D:\Apps\rea-video\verify-out\synth_ai\remake\jobs.local-h3.json
```

`rea-video remake-run --dry-run` 只打印任务，不会进入本通道。要看图，用上面的 adapter `dry-run --jobs ... --shot N`。如果 jobs 里仍是 `seedance2.0`，dry-run 会拒绝，避免把付费任务悄悄改掉。

## 最短的真实 GPU 冒烟命令

先确认 ComfyUI 已在 `127.0.0.1:8188` 监听。把 `FIRST` 和 `LAST` 换成合成任务里真实存在的两张图。输出必须在 capsule 的 `output` 目录下。这条命令不需要 VLM，也不会读取 `MINIMAX_API_KEY`。

```bat
set CAPSULE_LOCAL_H3_PROFILE=smoke
D:\Apps\capsule-cinema\.venv\Scripts\python.exe D:\Apps\capsule-cinema\lib\custom_tools\video_generation\local_comfy_minimax_h3_adapter.py generate --profile smoke --generation-type first_last_frame --aspect-ratio 9:16 --duration 4 --seed 5555 --prompt "主体从首帧自然过渡到尾帧" --first FIRST.png --last LAST.png --output D:\Apps\capsule-cinema\output\local_h3_smoke\video.mp4 --comfy-url http://127.0.0.1:8188
```

成功时命令退出码为 0，并且 `video.mp4` 是带 `ftyp` 的 mp4。云端代理没有在 FM 的 GPU 上跑过这条命令，不能把它的结果说成已经生成。

用 rea 跑同一条本地通道（仍是冒烟档）时，先 `rewrite-jobs`，再：

```bat
set CAPSULE_LOCAL_H3_PROFILE=smoke
rea-video remake-run D:\Apps\rea-video\verify-out\synth_ai\remake\jobs.local-h3.json --capsule-root D:\Apps\capsule-cinema --only 1
```

正式复刻不要加 `smoke`。去掉 `CAPSULE_LOCAL_H3_PROFILE`，或设成 `turbo`，再跑 `remake-run`。9:16、4 秒会使用 768×1344、107 帧、8 步和 FL2VA turbo LoRA。

## 回滚

1. 删除 `lib\custom_tools\video_generation\local_comfy_minimax_h3_adapter.py`。
2. 从 `local-channels\tool_registry.yaml` 和 `tool_capabilities.yaml` 去掉 `LocalComfyMiniMaxH3VideoGeneratorTool`。如果这两个文件里只有这一项，可以删除整个文件。
3. 若打过 rea 补丁：在 `D:\Apps\rea-video` 执行 `git apply -R rea-video-windows-python.patch`，或还原 `rea_video\cli.py`。
4. 不需要重启 ComfyUI，也不要卸载模型。已提交到 ComfyUI 队列里的任务可用 ComfyUI 界面取消；本通道超时只会删除自己的 prompt，并只在该 prompt 仍在运行时中断它。

## 限制

- 云端环境访问不到 FM。部署和第一次出片要在 FM 上做。
- 冒烟分辨率和 22 帧低于 H3 训练区间（约 124–362 帧）。它只验证上传、构图、队列、历史和 mp4 回拷。
- 官方蓝图使用 `BasicGuider`、`res_multistep`、`simple` 调度，不额外加 CFG 或 `MiniMaxH3SigmaShift`。
- 本机没有已验证的 FL2VA 工作流文件；图是按 ComfyUI `ab08f7b` 的 `Image to Video (MiniMax H3)` 蓝图和 `nodes_minimax_h3.py` 组的。
- `reference_to_video` 使用已验证的 Ref2VA 采样（euler；turbo 档 4 步并加载 4-step LoRA）。rea 的 `first_last_frame` 不会进入这条图。
- 文本编码器首次加载会占用大量内存。冒烟超时默认 30 分钟，turbo/quality 默认 60 分钟。
- 不支持云端节点、公网 Comfy、端口 7860，以及把 `minimax-h3` 当作本地引擎。
