[English](README.md) · **中文**

---

# MiniMax H3 在 8GB 笔记本上的六个实测陷阱

> 硬件：**RTX 5060 Laptop 8GB (sm_120) + 15.26 GiB 内存**，Windows 11 25H2，
> ComfyUI 0.35.0 (a7b1d39)，PyTorch 2.14.0+cu130，Python 3.13.12
> 本文所有数字均为该机器实测。**凡未实测者一律标注。**

这不是一篇「最佳参数推荐」。这是六个让我折腾了好几个小时的问题，每个都附可复现的方法 ——
而且**每一个我都先搞错过一次**。

---

## 一、10 秒的片子在 7 秒处悄悄丢掉主体 —— 是提示词，不是模型

**现象。** 10 秒 / 20 步 / 768×1024 的 I2VA，前 6 秒很好看，然后角色消失，最后 3 秒是空景。
更早在特写处推近，会得到**一块没有五官的空白皮肤**。

**根因。** 提示词只写了 `[Shot 1]` —— 一个镜头、没有时间线覆盖 —— 且运镜**没有落点**：

```
❌ 镜头以小幅度、慢速度向前推进              （只有幅度+速度，没有终点）
✅ The camera pushes in with small amplitude at slow speed
   toward her hand clutching the front of her sweater, and comes to rest at a medium shot.
```

官方指南（`references/base-en.txt`）的三条硬要求：

- **§4.3 运镜 = 类型 + 幅度 + 速度**，且官方示例内嵌落点（`toward the folded letter in her hands`）。
  只给幅度和速度，模型没有停止条件，就会一路推到主体出画。
- **§4.2** 后续镜头必须写 `[Shot N] At 00:0S.SSS, the camera cuts to ...`。
  只写 `[Shot 1]` 而片子有 10 秒，后段模型只能自己编。
- **§3.1** I2VA 结构是**首帧锚定 → 动作起始 → 持续发展 → 结果或反应**。原提示词停在「持续发展」。

同一句里还有两个错：

- 写「保持…与**整体构图不变**」**同时**要求推镜，**自相矛盾**。
  官方措辞是 `preserving her appearance, clothing, seat position, and the carriage layout`
  —— **不含构图**，构图必然会随运镜改变。
- 要求推近去看一张参考图里**故意用头发遮住的脸**，等于逼模型编造它没有的信息。

**修复效果。** 同 seed、同 20 步、同分辨率，**只改提示词**：

| | 修改前 | 修改后 |
|---|---|---|
| 7.0–10.0 秒 | ❌ 主体消失 | ✅ 主体全程在画面内 |
| 特写脸部 | ❌ 空白皮肤 | ✅ 侧脸轮廓，头发仍遮脸 |
| 推镜 | ❌ 失控推到极限特写 | ✅ 中景停住 |
| 6 秒切镜 | — | ✅ 精确命中 |

证据：`evidence/old_vs_new.png`、`evidence/dense_grid.png`（修改前）、`evidence/new_grid.png`（修改后）。
视频：`videos/01-...` 与 `videos/02-...`。

**结论：H3 出片后段劣化或丢主体，先怀疑提示词的时间线覆盖，再去动步数、采样器、加速节点。
20 步官方基线可以完整复现这个失败。**

---

## 二、sm_120 上 SageAttention 是负优化，而且它会让后端选择失效

**在 H3 真实几何下实测**（heads=56, head_dim=128, S=8192, bf16）：

| 后端 | 耗时 ms | 相对 L2 误差 |
|---|---|---|
| SDPA bf16（不加任何 flag 就是这个） | 143.1 | 0.0025 |
| **comfy-kitchen INT8** | **22.8** | **0.0164** |
| kitchen INT8 + `head_chunks=10` | 26.2 | 0.0164 |
| **KJ SageAttention（sm89 内核）** | **29.0** | **0.0387** |
| sage + `head_chunks=10` | 31.6 | 0.0387 |

**comfy-kitchen 比 SDPA 快 6.3 倍、比 sage 快 1.27 倍，误差还低 2.4 倍。**

更麻烦的是：`MiniMaxH3MemoryEfficientSageAttentionPatch` 直接调用 `_sageattn_int8_fp8_nhd`
（`ltxv_nodes.py:2106`），**完全绕过 `optimized_attention`/`wrap_attn`**。
只要图里挂着这个 patch，`ModelAttentionBackend` 和 `--use-ck-attention` 对 H3 的 50 个 block
**完全无效**。

**正确做法：摘掉 sage patch，加 `ModelAttentionBackend = "comfy kitchen attention"`。**

另外两条值得知道：

- `MiniMaxLowVRAMAttention` 和 sage patch **不冲突**。分块设置写在
  `transformer_options["minimax_head_chunks"]`（`minimax_nodes.py:188`），
  而 `minimax_sageattn_forward` 会主动读它（`ltxv_nodes.py:2102`），两种顺序结果一致。
  `head_chunks=10` 实测误差与不分块**逐位相同**，只慢 15%。
- CLI 的真名是 `--use-ck-attention`（不是 `--use-comfy-kitchen-attention`），
  六个 attention flag 在同一个 `add_mutually_exclusive_group()` 里。
  comfy-kitchen 的 CUDA 后端硬要求 cu130+（`comfy/quant_ops.py:22-28`）。

---

## 三、`SaveVideo` 的 H.264 重编码在奇数列/行尺寸下会失败 —— 用 `VHS_VideoCombine`

`SaveVideo` 有 `crf` 控件，但通过 API 够到它并不直观，而且**够到了也可能失败 —— 只要视频的宽或高是奇数**：

```
av.error.ExternalError: [Errno 542398533] Generic error in an external library:
    'avcodec_open2("libx264", {})'                    ← 连【空参数】都失败
    'avcodec_open2("libx264", {'crf': '12.0'})'
```

**原因是尺寸，不是编码器坏了。** `video_types.py` 把帧的宽高原样交给 libx264，同时设
`pix_fmt = "yuv420p"` —— 而 `yuv420p` 的色度是 2×2 下采样，**宽高必须都是偶数**。libx264
拒绝打开并返回 `EINVAL (22)`；又因为 PyAV 是在 `encode()` 里惰性打开的，最终只报一个
完全没提尺寸的外部库错误。

同一套 PyAV（18.1.0）、**不启动 ComfyUI** 实测：`64x64`、`1080x1920` 正常；`65x63`、
`63x65`、`65x65`、`1171x2532`、`1080x1441` 全部失败。我们的参考图是 **1259 × 1672**，
宽为奇数。已上报为 [ComfyUI #16544](https://github.com/Comfy-Org/ComfyUI/issues/16544)。

`format=auto` 不是替代方案：它的语义是「保留上游流、不重编码」，**改不了画质**。

这些嵌套下拉在 API 里的写法是**点号路径**（`comfy_api/latest/_io.py: finalize_prefix` 用 `.` 拼接）：

```json
"format": "mp4",
"format.codec": "h264",
"format.codec.encoding": "re-encode",
"format.codec.encoding.crf": 12.0
```

我先试错的三种写法，每一种都是**静默无效或直接报错**：

| 写法 | 结果 |
|---|---|
| 扁平键 `codec` / `encoding` / `crf` | **静默丢弃**（`crf=None`），输出逐字节相同 |
| 整个字典塞进 `format` | 校验器丢掉 `format` → `TypeError: missing required argument: 'format'` |
| 点号路径 | ✅ 解析成功，值到达 `avcodec_open2` → **然后 libx264 打不开** |

**解法：** `VHS_VideoCombine`（VideoHelperSuite）**调用外部 ffmpeg 可执行文件**，不走 PyAV。
它的 `crf` 实测可用 —— 同一段 48 帧输入：

```
crf=12  ->  570,872 字节
crf=19  ->  287,454 字节
crf=40  ->   40,296 字节
```

它还直接接受 `IMAGE` + `AUDIO`，所以 `CreateVideo` 可以删掉。
注意它的 `pix_fmt` / `crf` / `save_metadata` / `trim_to_audio` **不在 `/object_info` 里**
（挂在 `format` 组合框下），所以 UI→API 转换会丢掉它们。

**实测对比（同配置，10 秒 / 768×1024）：**

| 输出节点 | 码率 |
|---|---|
| `SaveVideo`（format=auto） | 2067 kb/s |
| **`VHS_VideoCombine`（crf=12）** | **10393 kb/s（×5.0）** |

探针脚本：`scripts/probe_encoder.py` —— **几秒出结果**，因为完全不加载模型
（只有 `LoadImage → RepeatImageBatch → 输出节点`）。**这个套路强烈推荐**，用来摸输出节点很快。

---

## 四、turbo LoRA 在 pruned 基座上只有 80.3% 生效

统计 LoRA 模块：**共 259 个，能应用 208 个（80.3%），不能应用 51 个（19.7%）**。
51 个全是 `adaln_proj.linear`，形状说明了原因：

```
pruned 基座    adaln_proj 输入维 = 8
LoRA（非pruned） adaln_proj 输入维 = 2688
```

LoRA 元数据自己写着：
`incompatible_base: MiniMax-H3 pruned_* (AdaLN input is 8, LoRA input is 2688)`。

**AdaLN 是时间步调制通路，而步数蒸馏 LoRA 要改的正是这里。所以蒸馏只被部分应用。**

这个失败每步刷 50 条报错：

```
ERROR lora diffusion_model.blocks.N.adaln_proj.linear.weight shape '[96768, 8]'
      is invalid for input of size 260112384
```

**它不是「进度提示」，是每步 50 个 LoRA 层应用失败。** 因为不中断运行，所以很容易被忽略。

复现：`scripts/lora_applicability.py`

---

## 五、卡住你的是 16GB 内存，不是 8GB 显存

| ComfyUI 会话状态 | 每 block | 10 秒成片 |
|---|---|---|
| **刚重启** | **0.9–6 秒** | **34–44 分钟** |
| 连续运行约 1.7 小时后 | **212 秒** | **约 12 小时** |

**同样参数。** 判断方法只有一个数字：**`nvidia-smi` 的功耗**。

- 正常采样：**64–98 W**
- 在换页：**平线 34–36 W**，而 `utilization.gpu` 仍显示 99%（这个计数器把搬运内存的核也算进去）

**循环采样功耗，一旦变平线就停 —— 这个信号让我在 5 分钟内砍掉了 9:16 那次尝试，而不是等几小时。**

支撑数据：

- **降分辨率不是杠杆**：一张 12GB 卡的一手测试把工作量缩小 40 倍，峰值显存只降 **0.5%**，内存降 1.0%。
- **容量由「分辨率 × 时长」决定，不是步数**：4060 8GB 上 1024×576/243f 跑 20 步能完（2762.89 秒），
  而 1344×768/**362f** 跑 **1 步** 就 CUDA OOM。
- **官方限制**：短边 768、面积 ≤ 768×1344、32 的倍数。3:4 下封顶 768×1024 = 0.786 MP；
  9:16 下是 768×1344 = 0.984 MP。
  ⚠️ **`ResolutionSelector` 的「百万像素」是个陷阱** —— 同一个 MP 值在不同比例下短边不同
  （3:4 设 1.0 MP 会得到 896×1184，短边 896，**超出训练区间**）。
- 在这台 16GB 机器上试 9:16（768×1344，+31% 像素）**直接掉进换页**（平线 34W），
  而几分钟前同样的图在 3:4 下跑得好好的。**+31% 像素就是那条线。**

相关：16GB 机器上 **`--disable-pinned-memory` 是必需的** ——
否则 `MAX_PINNED_MEMORY = ram * 0.90` 会让进程被系统杀掉。

---

## 六、8 步 vs 20 步：构图几乎一样，时间差 2.5 倍

同 seed、同提示词、同 768×1024，都使用第一节修好的提示词：

| | 20 步 | 8 步 |
|---|---|---|
| 相机计划 | 推近 → 6 秒切镜 → 肩颈侧脸 | **完全一致** |
| 0–10 秒主体 | ✅ | ✅ |
| 特写处织物纹理 | 略多 | 略柔和 |
| **墙钟（10 秒片）** | **33.8–43.5 分钟** | **14.3 分钟** |
| 码率 | 2067 kb/s（SaveVideo auto） | 10393 kb/s（VHS crf=12） |

**8 步用 1/2.5 的时间，拿到正常观看尺寸下基本相同的画面。**

**必须说清的偏差**：这两次用了**不同的编码器**，而且 8 步那版有 **5 倍码率却仍然略软**。
这算是「多出的步数确实买到了一点真实细节」的弱证据。

对比图：`evidence/steps_8_vs_20.png`

**实用结论：这台机器上 10 秒 / 768×1024，8 步适合作日常档，20 步留给最终成片。**

---

## 耗时模型（仅在 5 秒 / 768×1024 / kitchen / LoRA 1.0 下有效）

```
T_wall ≈ 102 + 34.1 × 步数         四点回代残差 ≤ 3.2%（4/6/8/20 步）
```

实测：4 步 245.3 秒 · 6 步 297.4 秒 · 8 步 377.9 秒 · 20 步 785.3 秒。

**⚠️ 不要拿它外推到更长的片子。** 10 秒档每步约 95 秒，而不是 5 秒斜率预测的约 57 秒 ——
我在这上面搞错过，预测 28.7 分钟，实际跑了 38–43.5 分钟。
10 秒 / 20 步共四次实测：**2613、2283、2560、2029 秒 → 33.8–43.5 分钟，±11%。**

一个有意义的推论（5 秒档）：**20 步 / 4 步 = 3.20×，不是 5×** ——
因为存在约 **102 秒固定开销**，它占 4 步运行的 42%，占 20 步的只有 13%。

---

## 仓库结构

```
scripts/
  probe_encoder.py        # 秒级探针：SaveVideo vs VHS 的 crf（不加载模型）
  attention_probe.py      # SDPA / kitchen / sage 在 H3 几何下的耗时与误差
  lora_applicability.py   # 逐 key 比对 LoRA 与 pruned 基座形状
  build_workflow.py       # 用声明式节点表生成 ComfyUI 工作流
  benchmark_loop.py       # 重启 → 提交 → 采样峰值 → 解析分段 → 写 CSV
workflows/
  H3_Quality.json   20 步 · 无 LoRA · res_multistep   （官方基线）
  H3_Balanced.json   8 步 · turbo LoRA @1.0 · euler    （推荐日常档）
  H3_Fast.json       4 步 · turbo LoRA @1.0 · euler
evidence/
  results.csv        # 14 次运行 × 33 字段（峰值显存/内存、分段耗时、崩溃标记）
  old_vs_new.png     # 提示词修复对照（同 seed、同时间点）
  dense_grid.png     # 7.0 秒主体消失的证据
  steps_8_vs_20.png  # 8 步 vs 20 步
  encoder_isolation.png  # 同一批帧压到两种码率的对照
videos/              # 三段成片（修前 / 修后 20 步 / 修后 8 步高码率）
docs/
  working-report-cn.md   # 过程稿，内含 5 条被推翻的结论，仅供追溯
```

## 诚实的完成度说明

**已实测：** 第一至六节每条都有上面的复现方式；其中 1、3、4 用同 seed 单变量或逐字节对比确认过。

**未验证 / 待办：**

- **RTX 4060 Laptop (sm_89) 已有实测**（2026-09）：4 步 / 5 秒 / 768×1024 下
  pytorch attention 403.8 s · comfy-kitchen 270.1 s · sage 263.0 s。
  **kitchen 在 4060 上同样比 pytorch 快 1.5×**，所以本建议无需按架构区分。
  仍未测：4060 上的 8 步 / 20 步耗时。
  那台机器的提交上限只有 22.14 GB（比本机低 29%），A1 的绝对值可能被页面文件扩张污染。
- **不声称附带的三个工作流是最优解。** 它们是有证据支撑的合理起点。
- 第六节的 8 vs 20 对比**被编码器差异污染**，且只有一个 seed。
- `ref_image_size` 有个 `max` 选项，节点自己的 tooltip 说能给出「最好的身份保真度」但「可能慢好几倍」。**本项目未测。**
- 9:16（768×1344）是因为**内存不足**而放弃，不是因为它错 —— 页面文件更大的话它可能是更好的选择。
- 长片：测过的都是 5 秒或 10 秒。官方训练区间约 124–362 帧，这里没有超过 243 的探针。
- **本会话中我至少 5 次下了错结论又推翻**（Sol-Attn「无损」、TE-Speed 证据张冠李戴、
  三次错误的 CRF「已验证」、一次耗时外推、一次崩溃归因）。
  **凡本文未明确标「实测」的句子，都请当作可疑。**

## 许可

脚本 MIT。工作流 JSON 是生成产物。**不含也不分发任何模型权重、LoRA 文件或自定义节点代码。**
