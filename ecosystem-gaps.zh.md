# MiniMax H3 开源生态缺口清单

**2026-10-24** · 数据来自 GitHub API 的当日扫描 · 口径与局限见文末

## 一句话

H3 的社区不缺"怎么跑起来"的教程，缺的是"跑不起来时先查哪里"。相关仓库里超过 30 天没有任何更新的 issue 共 57 条，其中 41 条从未被任何人回复。

## 现状

| 仓库 | 开着的 | ≥30 天无更新 | 其中零回复 |
|---|---|---|---|
| `MiniMax-AI/awesome-minimax-h3-integration` | 0 | 0 | 0 |
| `wildminder/awesome-minimax-H3` | 0 | 0 | 0 |
| `Larryvrh/ComfyUI-MiniMax-H3-Turbo`（588★） | 23 | 19 | 15 |
| `SkyNotSilent/awesome-MiniMax-H3-cases` | 2 | 1 | 0 |
| `MiniMax-AI/MiniMax-H3`（9.2k★） | 54 | 38 | 26 |
| `Comfy-Org/ComfyUI`（标题含 minimax） | 219（含 PR） | — | — |

两个整理库零积压，压力全在社区 Turbo 包和官方参考实现仓库两端。ComfyUI 侧的另一半情况：标题带 minimax 的条目最近一周仍有新 PR（#16713 一天前、#16933 五天前、#16889 十一天前），问题不缺，缺的是归类。

## 三类问题

**一、装不上、跑不起来**（数量最多，答案高度重复）

- Turbo 包：托管环境装不了自定义节点（#7）、ComfyUI 更新后 Turbo LoRA loader 失效（#17）、Apple silicon 不支持（#14）、`apply_lora` 的 `AttributeError`（#11）
- 官方仓库：diffusers 推理直接失败（#4）、依赖版本对不上（#37）、异步任务未轮询导致下载失败（#38）、权重下载权限（#20）

**二、LoRA 与 adaLn 行数对不上**（技术性最强，几乎没有回应）

- 社区 LoRA 在 pruned int8 基座上静默失败，键名多一层 `diffusion_model.` 前缀（#28）
- Turbo LoRA 的音轨条件行数不匹配（#19、#25）；keyframe 带 `audio_latent` 时 `_unique_t` 漏掉 audio 行（#30）
- MPS 后端上 adaLn 注入产生 NaN / 黑帧（#26）

这一类的特点是：有人已经写了修复（#19、#25、#26 都有对应 PR），但都躺在那里没人合。

**三、显存与性能**

- VAE tile 参数是否可配置（#46）
- SGLang 本地部署时 0% GPU、进度不动（#42）
- ComfyUI 侧：fp16 推理支持（#16508）、显存估算（#15983）、packed-sequence 显存准入（#16460）

## 文档缺口

1. 没有一页"最低显存配置"：8 GB 能跑到什么、需要哪些启动参数、什么情况下该放弃。
2. 没有"常见报错"索引：上面第一类的问题，答案其实只有三四种。
3. 没有可复用的最小工作流示例：ComfyUI 侧大量条目是"跑不出结果"，缺一个能照抄的起点。

## 我能补的那一条

8 GB 单卡的三档工作流（20 / 8 / 4 步）和六个陷阱的逐条复现脚本已经公开，其中两份整理被合并进了 `MiniMax-AI/awesome-minimax-h3-integration`（#19）和 `wildminder/awesome-minimax-H3`（#7）。

能补的是第一类：把"跑不起来"的报错归成几条判据，写成可复现的索引，并附上 534 条、10 秒粒度的功耗 / 显存 / 时钟采样作为判定依据。

这份清单不是求职信。如果这块已经有人在推进，欢迎指正，或者直接拿去用。

## 口径与局限

- 统计时间 2026-10-24，GitHub REST API；统计对象为 `state=open` 且最近更新距今 ≥30 天的 issue / PR。
- "零回复"指 `comments = 0`。
- 41 条里至少 5 条是无内容帖（`#23`「動画」、`#63`「b」、`#72`、`#76`、`#78`），去掉后需要处理的技术帖约 36 条。
- ComfyUI 只统计标题含 minimax 的搜索结果，未逐条阅读，因此那一行只作规模参考。
