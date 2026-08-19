# MiniMax-H3 on BW1100

本文给出 MiniMax-H3 在 BW1100（144 GiB/卡）上的推荐部署方式，覆盖 T2VA、FL2VA、Ref2VA，以及 1/2/4/8 卡的最佳实践。

## 模型与场景

| 场景 | `model-variant` | 输入 | 输出 |
| --- | --- | --- | --- |
| T2VA | `fl2va` | 文本 | 视频 + 音频 |
| FL2VA | `fl2va` | 文本 + 首尾帧/关键帧 | 视频 + 音频 |
| Ref2VA | `ref2va` | 文本 + 参考图像、音频或视频 | 视频 + 音频 |

模型使用 BF16 精度。T2VA 与 FL2VA 可以共用一个 `fl2va` Server；切换到 Ref2VA 时必须停止当前 Server，以 `ref2va` 重新启动。

| 模型权重 | 精度 | SGLang 版本 | 推荐硬件 |
| --- | --- | --- | --- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.15 | BW1100 |

## 运行镜像

```text
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang-0.5.15-minimax-h3-fixed-cachedit-0817
```

镜像已包含 SGLang 0.5.15、MiniMax-H3 CacheDiT 修复和调优 hipBLASLt。不要再覆盖安装 SGLang、FlashAttention 或 `sglang-kernel`。


## 卡数与推荐布局

### 不开启 CacheDiT

| 卡数 | 推荐布局 | `NUM_GPUS` | `TP_SIZE` | `SP_DEGREE` | `ULYSSES_DEGREE` | 说明 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | TP1 | 1 | 1 | 1 | 1 | 单卡常驻，无 offload |
| 2 | TP2 | 2 | 2 | 1 | 1 | 性能优于 SP2，显存也更低 |
| 4 | TP2 + SP2 | 4 | 2 | 2 | 2 | 推荐的性能/显存平衡布局 |
| 8 | SP8 | 8 | 1 | 8 | 8 | 当前无缓存正式结果中最快 |

### 开启 CacheDiT

| 卡数 | 推荐布局 | `NUM_GPUS` | `TP_SIZE` | `SP_DEGREE` | `ULYSSES_DEGREE` | 说明 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | TP1 | 1 | 1 | 1 | 1 | 单卡最佳 |
| 2 | TP2 | 2 | 2 | 1 | 1 | 明显快于 SP2 |
| 4 | TP2 + SP2 | 4 | 2 | 2 | 2 | 明显快于 SP4 |
| 8 | TP8 | 8 | 8 | 1 | 1 | 当前最快且单卡显存最低 |

## 启动 Server

先按上表选择卡数与布局。以下示例是八卡无 CacheDiT 的 SP8；其他卡数只需替换对应变量。

```bash
export GPU_IDS=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export TP_SIZE=1
export SP_DEGREE=8
export ULYSSES_DEGREE=8

export MODEL_PATH=/models/MiniMax-H3
export OUTPUT_PATH=/workspace/outputs

# T2VA、FL2VA 使用 fl2va；Ref2VA 改成 ref2va。
export MODEL_VARIANT=fl2va
# export MODEL_VARIANT=ref2va

export PORT=30010
export SCHEDULER_PORT=30011
export MASTER_PORT=30012

export MINIMAX_H3_TORCH_SDPA_BACKEND=flash
export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1


HIP_VISIBLE_DEVICES="$GPU_IDS" sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --model-variant "$MODEL_VARIANT" \
  --num-gpus "$NUM_GPUS" \
  --tp-size "$TP_SIZE" \
  --sp-degree "$SP_DEGREE" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode "$WARMUP_MODE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --scheduler-port "$SCHEDULER_PORT" \
  --master-port "$MASTER_PORT" \
  --strict-ports \
  --output-path "$OUTPUT_PATH"
```

八卡开启 CacheDiT 时，将布局变量改为 TP8：

```bash
export NUM_GPUS=8
export TP_SIZE=8
export SP_DEGREE=1
export ULYSSES_DEGREE=1
```

`MINIMAX_H3_TORCH_SDPA_BACKEND=auto` 只控制 Video VAE 内部 PyTorch SDPA；DiT 主干由 `--attention-backend fa` 选择 HCU FlashAttention。稳定部署不建议强制 Video VAE Flash。

## 开启 CacheDiT

CacheDiT 是近似加速。先设置以下变量，再执行同一条 Server 命令：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3

# 建议验证阶段开启，检查命中和残差是否正常。
export SGLANG_MINIMAX_H3_CACHE_DIT_STATS=true
```

推荐配置的全称是：`Front Blocks=1`、`Back Blocks=0`、`Residual Diff Threshold=0.24`、`Max Continuous Cached Steps=3`。正式日志应看到 cache hit，并确认 `residual_diffs` 中没有 `NaN`。本文性能结果的典型命中是 33/49 个 Denoising step。

## Warmup

Warmup 不能省略，否则首个正式请求会混入 kernel、通信和内存初始化开销。

- 在线部署：使用 `--warmup-mode server`。Server 会在 ready 前自动执行一次 2-step 请求；warmup 期间 `/health` 可能返回 503。
- 严格性能测试：使用 `--warmup-mode off`，依次执行 3 次 2-step 短请求；每次等 Server 日志出现 `Pixel data generated successfully` 后再发送下一次。随后第 4 次发送 50-step 正式请求。

下面是 2-step T2VA warmup 请求。执行三次即可：

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside.\noverall_soundscape: Quiet indoor ambience.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 2,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

## 正式请求

### T2VA

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside, soft indoor lighting, gentle ambient room tone.\noverall_soundscape: Quiet indoor ambience with soft snowfall.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

### FL2VA

FL2VA 与 T2VA 共用 `fl2va` Server。首帧使用 `frame_index=0`，尾帧使用 `frame_index=-1`。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "fl2va",
    "prompt": "At 0.00 seconds, <Picture 1> is fully referenced. Generate a smooth cinematic camera orbit while preserving the subject and scene.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/fl2va_first_frame.png",
        "role": "keyframe",
        "frame_index": 0
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 2101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

### Ref2VA

先停止 `fl2va` Server，把 `MODEL_VARIANT` 改为 `ref2va`，再重新启动。参考素材路径必须能在容器内访问。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "ref2va",
    "prompt": "<Subject 1> is the subject in <Picture 1>. <Audio 1> is the voice reference. Create a realistic video with precise lip sync while preserving the subject appearance.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/ref2va_image.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "/inputs/ref2va_audio.mp3",
        "role": "reference"
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 3101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

接口先返回 `queued` 和任务 ID。日志出现 `Output saved to ...mp4` 与 `Pixel data generated successfully` 后，视频位于 `--output-path` 指定目录。

## 显存不足时的处理

BW1100 的 T2VA 推荐布局无需 offload。更长视频或 Ref2VA 如果 OOM，按以下顺序处理：

1. 先将 `--text-encoder-cpu-offload` 改为 `true`。
2. 仍不足时，再尝试 `--use-fsdp-inference true`。
3. FSDP 或 DiT layerwise offload 作为显存兜底时，先关闭 CacheDiT；本文不推荐将它们组合使用。

## 日志检查

```text
Using HCU FlashAttention-2 backend on HCU.
Using fa attention backend
Pipeline instantiated
[MiniMaxH3TextEncodingStage] finished in ... seconds
[MiniMaxH3DenoisingStage] finished in ... seconds
[MiniMaxH3DecodingStage] finished in ... seconds
Output saved to ...mp4
Pixel data generated successfully in ... seconds
```

开启 CacheDiT 后还要确认命中数大于 0，并确认 `residual_diffs` 没有 `NaN`。不同节点负载、温度和频率会带来小幅波动，正式对比必须使用相同请求与 warmup 口径。

## 性能数据

### 测试口径

- T2VA，5 秒，768p，50 inference steps；日志实际执行 49 次 Denoising iteration。
- 3 次 2-step 短 warmup，第 4 次 50-step 正式测量。
- `Server Total` 是 Server 返回的生成耗时；`s/step = Denoising / 49`。
- CacheDiT 使用 `Fn=1/Bn=0/WARMUP=4/RDT=0.24/MC=3`，命中 33/49，残差无 `NaN`。
- CacheDiT 是近似优化，不能把收益写成无损加速。

### 不开启 CacheDiT

以下均为 NMZ26/BW1100、SGLang 0.5.15、不开启 CacheDiT、启用镜像内 hipBLASLt 库并设置 `MINIMAX_H3_TORCH_SDPA_BACKEND=flash` 的正式结果。

| 卡数 | 布局 | Server Total | Denoising | s/step | Decoding | 单卡峰值显存 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TP1 | 939.68s | 785.93s | 16.0393s | 100.41s | 133,176 MiB（90.33%） |
| 2 | SP2 | 493.79s | 425.99s | 8.6937s | 51.47s | 111,077 MiB（75.34%） |
| 2 | TP2 | **489.83s** | 421.78s | 8.6078s | 49.79s | 77,547 MiB（52.60%） |
| 4 | SP4 | **260.02s** | 225.61s | 4.6043s | 25.61s | 98,450 MiB（66.77%） |
| 4 | TP2 + SP2 | 266.43s | 233.90s | 4.7735s | 25.45s | 66,657 MiB（45.21%） |
| 8 | SP8 | **133.45s** | 113.02s | 2.3066s | 15.84s | 91,741 MiB（62.22%） |
| 8 | TP2 + SP4 | 138.46s | 119.94s | 2.4478s | 14.73s | 60,280 MiB（40.88%） |
| 8 | TP4 + SP2 | 157.32s | 138.42s | 2.8248s | 14.55s | 143,099 MiB（97.06%） |
| 8 | TP8 | 162.34s | 141.41s | 2.8866s | 14.52s | 36,137 MiB（24.51%） |

### 开启 CacheDiT：完整布局矩阵

| 卡数 | 布局 | Server Total | Denoising | s/step | Decoding | 单卡峰值显存 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | TP1 | 387.61s | 269.71s | 5.5042s | 100.83s | 126,178 MiB（85.58%） |
| 2 | SP2 | 357.19s | 274.29s | 5.5978s | 65.58s | 105,804 MiB（71.76%） |
| 2 | TP2 | **200.46s** | 141.89s | 2.8958s | 49.49s | 74,052 MiB（50.23%） |
| 4 | SP4 | 188.21s | 144.44s | 2.9477s | 34.47s | 90,090 MiB（61.10%） |
| 4 | TP2 + SP2 | **108.20s** | 77.66s | 1.5848s | 25.49s | 57,880 MiB（39.26%） |
| 8 | SP8 | 96.20s | 72.48s | 1.4791s | 18.93s | 84,458 MiB（57.28%） |
| 8 | TP2 + SP4 | 99.07s | 75.31s | 1.5370s | 18.75s | 52,156 MiB（35.37%） |
| 8 | TP4 + SP2 | 107.58s | 83.42s | 1.7025s | 18.70s | 37,060 MiB（25.14%） |
| 8 | TP8 | **66.28s** | 48.26s | 0.9848s | 14.79s | 29,072 MiB（19.72%） |

结论：单卡选择 TP1；双卡选择 TP2；四卡不开 CacheDiT 选择 SP4、开启 CacheDiT 选择 TP2 + SP2；八卡不开 CacheDiT 选择 SP8、开启 CacheDiT 选择 TP8。
