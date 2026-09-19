# MiniMax H3 — 8 GB 显存工作流（三档）

三档可直接运行的 ComfyUI 工作流，针对 **8 GB 显存的笔记本 GPU** 调过。
每个参数的选择都有实测依据，出处见文末。

| 文件 | 步数 | sampler | LoRA | TE-Speed | 10 秒成片实测 |
|---|---|---|---|---|---|
| `H3_Quality.json` | 20 | `res_multistep` | 无 | 关 | **33.8–43.5 分钟** |
| **`H3_Balanced.json`** | **8** | `euler` | turbo @1.0 | 关 | **14.3 分钟** |
| `H3_Fast.json` | 4 | `euler` | turbo @1.0 | 关 | 约 9 分钟 |

**日常用 Balanced。** 同 seed 对比下，8 步和 20 步的构图基本一致，时间只有 1/2.5。

---

## 1. 你需要先准备这些模型

放到 ComfyUI 对应的 `models/` 子目录下：

| 用途 | 文件名 | 放哪里 |
|---|---|---|
| 主模型 | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| 视频 VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `models/vae/` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |
| turbo LoRA | `minimax_h3_turbo_v4_step600_comfyui_T8-convert.safetensors` | `models/loras/` |

> **量化格式说明**：主模型优先用 `int8_convrot`（需要 PyTorch cu130）；用不了再退 `fp8_scaled`。
> 文本编码器的 `nvfp4_awq` **不需要 Blackwell 显卡**，而且比 `int8` 版本小 10.7 GiB ——
> 对 16 GB 内存的机器这是关键差别。
>
> **Quality 档不带 LoRA**，所以那份工作流不需要 LoRA 文件。

### 必须装的自定义节点

| 节点 | 来源 | 用途 |
|---|---|---|
| `MiniMaxLowVRAMAttention` / `MiniMaxChunkFeedForward` | ComfyUI 内置 | 降显存（源码级无损） |
| `ModelAttentionBackend` | ComfyUI 内置 | 切到 comfy-kitchen 注意力后端 |
| `VHS_VideoCombine` | **ComfyUI-VideoHelperSuite** | 输出视频（见下方说明） |
| `MiniMaxH3SigmaShift` | ComfyUI 内置 | 显式控制 shift |

---

## 2. 启动参数

```
--disable-pinned-memory        ← 16 GB 内存机器【必须】
```

**不要加**：
- `--use-sage-attention` —— 在 sm_120 上比 kitchen 慢且误差大 2.4 倍，而且会让后端选择失效
- `--use-flash-attention` —— sm_120 上不可用
- `--fast` / `--fp16-unet` —— H3 的计算 dtype 是 bf16，没有 fp16，会出全黑帧

---

## 3. 怎么用

1. 把 json 拖进 ComfyUI 窗口
2. **把 `LoadImage` 换成你自己的图**（默认指向 ComfyUI 自带的 `example.png`，只是个占位）
3. **把提示词换成你自己的** —— 但**保持这个结构**：

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] …… The camera pushes in with small amplitude
at slow speed toward <具体落点>, and comes to rest at a medium shot. …… [Shot 2] At 00:06.000,
the camera cuts to ……

overall_soundscape: ……

non_diegetic_music: ……
```

**三个最容易踩的坑（都实测过）：**

| 坑 | 后果 |
|---|---|
| 10 秒的片子只写 `[Shot 1]` | 后段没有内容可描述，**模型会自己填空景**，主体消失 |
| 运镜只写幅度和速度、**没有落点** | 模型不知道推到哪停，**一路推过头把主体推出画面** |
| 同时写「保持构图不变」和「推镜」 | **自相矛盾**，构图必然随运镜改变 |

4. 每次跑之前 **重启 ComfyUI**（见下方第 5 节）
5. 运行

---

## 4. 分辨率：不要动那个「百万像素」数字

官方硬约束：**短边 ≤ 768、面积 ≤ 768×1344、32 的倍数**。

`ResolutionSelector` 的「百万像素」是**陷阱** —— 同一个 MP 在不同比例下短边不同：

| 比例 | 短边=768 时应填的 MP | 实际输出 |
|---|---|---|
| 1:1 | 0.5625 | 768×768 |
| **3:4（默认）** | **0.75** | **768×1024** |
| 4:3 | 0.75 | 1024×768 |
| 9:16 | 0.959 | 768×1344 |
| 16:9 | 0.959 | 1344×768 |
| 2:3 | 0.844 | 768×1152 |

**填 1.0 MP 配 3:4 会得到 896×1184 —— 短边 896 超出训练区间，画质反而更差。**

> 9:16 的 768×1344（0.984 MP）是**合法上限**，但在 16 GB 内存的机器上实测会掉进换页
> （GPU 功耗从 80 W 掉到平线 34 W）。内存够大再试。

---

## 5. ⚠️ 每次运行前必须重启 ComfyUI

这是**收益最大的单条建议**，远超任何节点级优化：

| ComfyUI 会话状态 | 每 block | 10 秒成片 |
|---|---|---|
| **刚重启** | **0.9–6 秒** | **34–44 分钟** |
| 连续运行约 1.7 小时后 | **212 秒** | **约 12 小时** |

**怎么判断自己在掉速**：看 `nvidia-smi` 的**功耗**。

- 正常采样：**64–98 W**
- 在换页：**平线 34–36 W**，而利用率仍显示 99%（那个计数器把搬运内存的核也算进去）

**循环采样功耗，一旦变平线就停 —— 别等。**

---

## 6. 关于输出节点为什么用 `VHS_VideoCombine` 而不是 `SaveVideo`

`SaveVideo` 有 `crf` 控件，但在这套环境里**它的重编码通道是坏的**：

```
avcodec_open2("libx264", {})   ← 连【空参数】都失败
```

PyAV 自带的 libx264 打不开，所以 `SaveVideo` 唯一能用的模式是 `format=auto`，
而它的语义是「保留上游流、不重编码」——**改不了画质**，实测码率被压在 2067 kb/s。

`VHS_VideoCombine` 调用外部 ffmpeg 可执行文件，**crf 实测可用**：

| 输出节点 | 同一条片子的码率 |
|---|---|
| `SaveVideo`（auto） | 2067 kb/s |
| **`VHS_VideoCombine`（crf=12）** | **10393 kb/s（×5.0）** |

crf 越小画质越高、文件越大。**12 ≈ 视觉无损**；想让文件小一点，改到 18–20 也够用
（同一段测试输入：crf=12 → 570 KB，crf=19 → 287 KB，crf=40 → 40 KB）。

---

## 7. 已知限制

- **只在 RTX 5060 Laptop 8 GB (sm_120) + 15.26 GiB 内存上实测过。**
  **4060 用户请自己重测注意力后端** —— 它是 sm_89，FlashAttention 可用，
  不能假设 comfy-kitchen 在那上面也最优。
- 测过的时长只有 5 秒和 10 秒。官方训练区间约 124–362 帧。
- 三档**都不使用 TE-Speed** —— 实测它只值 18%，而代价是闭源不可审计的有损项。
- 这些工作流不是「最优解」，是**有证据支撑的合理起点**。

---

## 8. 依据出处

每条参数选择背后都有实测，完整证据、脚本和原始数据在：

**https://github.com/FlowForgotLab/h3-8gb-traps**

包含：kitchen vs sage vs SDPA 的注意力对照、LoRA 80.3% 应用率的逐 key 比对、
输出节点探针（不加载模型，秒级出结果）、14 次运行的 33 字段 CSV。
