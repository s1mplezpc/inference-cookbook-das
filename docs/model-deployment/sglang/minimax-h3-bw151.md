# MiniMax-H3 on BW151

本文给出 MiniMax-H3 在 BW151（64 GiB/卡）上的推荐部署方式，覆盖 T2VA、FL2VA、Ref2VA，以及 2/4/8 卡的最佳实践。单卡无法稳定容纳完整模型，不作为推荐配置。

## 模型与场景

| 场景 | `model-variant` | 输入 | 输出 |
| --- | --- | --- | --- |
| T2VA | `fl2va` | 文本 | 视频 + 音频 |
| FL2VA | `fl2va` | 文本 + 首尾帧/关键帧 | 视频 + 音频 |
| Ref2VA | `ref2va` | 文本 + 参考图像、音频或视频 | 视频 + 音频 |

模型使用 BF16 精度。T2VA 与 FL2VA 可以共用一个 `fl2va` Server；切换到 Ref2VA 时必须停止当前 Server，以 `ref2va` 重新启动。

| 模型权重 | 精度 | SGLang 版本 | 推荐硬件 |
| --- | --- | --- | --- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.15 | BW151 |

## 运行镜像

```text
docker pull harbor.sourcefind.cn:5443/dcu/admin/base/custom:sglang-0.5.15-minimax-h3-fixed-cachedit-0817
```

镜像已包含 SGLang 0.5.15、MiniMax-H3 CacheDiT 修复和 BW151 调优 hipBLASLt。不要再覆盖安装 

## 卡数与推荐布局

### 性能优先配置

| 卡数 | 推荐布局 | `NUM_GPUS` | `TP_SIZE` | `SP_DEGREE` | `ULYSSES_DEGREE` | Text Encoder offload | 说明 |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 2 | TP2 | 2 | 2 | 1 | 1 | `true` | 只 offload Text Encoder；其余 offload 与 FSDP 关闭 |
| 4 | TP2 + SP2 | 4 | 2 | 2 | 2 | `false` | 性能最快，但峰值可达 92.55% |
| 8 | TP2 + SP4 | 8 | 2 | 4 | 4 | `false` | 当前综合性能最佳 |

四卡如需更多显存余量，可改为 TP4：`TP_SIZE=4`、`SP_DEGREE=1`、`ULYSSES_DEGREE=1`。八卡如需更多显存余量，可改为 TP8：`TP_SIZE=8`、`SP_DEGREE=1`、`ULYSSES_DEGREE=1`。

CacheDiT OFF 与 ON 均使用上表布局。CacheDiT 只改变是否复用 Denoising block，不改变并行参数。

## 启动 Server

先按卡数从上表设置变量。以下示例是八卡 TP2 + SP4；双卡或四卡只需替换对应变量。

```bash
export GPU_IDS=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export TP_SIZE=2
export SP_DEGREE=4
export ULYSSES_DEGREE=4
export TEXT_ENCODER_OFFLOAD=false

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
  --text-encoder-cpu-offload "$TEXT_ENCODER_OFFLOAD" \
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

BW151 双卡必须开启 Text Encoder CPU offload。其他布局如果 OOM，按以下顺序处理：

1. 先将 `--text-encoder-cpu-offload` 改为 `true`。
2. 仍不足时，再尝试 `--use-fsdp-inference true`。
3. FSDP 或 DiT layerwise offload 作为显存兜底时，先关闭 CacheDiT；本文不推荐将它们组合使用。

Ref2VA 的参考素材编码显存压力高于 T2VA/FL2VA，可能需要比表中 T2VA 推荐配置更保守的 offload。

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

| 卡数 | 布局 | Offload | Server Total | Denoising | s/step | Decoding | 单卡峰值显存 | 备注 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2 | TP2 | Text Encoder | 1111.33s | 1040.87s | 21.2422s | 24.94s | 43,270 MiB（66.04%） |  |
| 4 | TP2 + SP2 | 全关 | 266.94s | 253.05s | 5.1643s | 6.58s | 59,616 MiB（90.97%） | |
| 8 | TP2 + SP4 | 全关 | 139.45s | 130.53s | 2.6639s | 4.01s | 52,814 MiB（80.59%） |  |

### 开启 CacheDiT：完整布局矩阵

| 卡数 | 布局 | Offload | Server Total | Denoising | s/step | Decoding | 单卡峰值显存 |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | TP2 | Text Encoder | 396.13s | 346.37s | 7.0687s | 24.76s | 44,242 MiB（67.52%） |
| 4 | TP2 + SP2 | 全关 | **88.86s** | 77.50s | 1.5816s | 6.13s | 60,638 MiB（92.55%） |
| 4 | TP4 | 全关 | 97.27s | 85.05s | 1.7358s | 6.61s | 44,796 MiB（68.37%） |
| 8 | TP2 + SP4 | 全关 | **50.42s** | 44.00s | 0.8979s | 3.54s | 53,208 MiB（81.21%） |
| 8 | TP4 + SP2 | 全关 | 80.05s | 71.85s | 1.4663s | 3.69s | 37,666 MiB（57.49%） |
| 8 | TP8 | 全关 | 76.86s | 69.31s | 1.4145s | 3.37s | 30,582 MiB（46.68%） |

结论：双卡选择 TP2 + Text Encoder offload；四卡性能优先选择 TP2 + SP2、显存优先选择 TP4；八卡性能优先选择 TP2 + SP4、显存优先选择 TP8。
