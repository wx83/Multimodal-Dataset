# Worker 契约：这条 pipeline 真正的换模型入口

这份文档回答一个具体问题：**「他们可以自己换 model」到底怎么换。**

这套东西的定位是提供架构而非追某个模型的极致效果，那么「怎么换」就是产品说明书
本身。而截至 2026-08-30，这个契约只存在于六个 worker 的平行实现里，没有写下来过。

## 扩展点是子进程契约，不是类名

每个重模型都住在自己的 conda env 里，由 `models/<name>_model.py` 以**子进程**方式
调用 `models/<name>_worker.py`。这条边界就是换模型的地方——它同时隔离了依赖冲突
（SAM3 要 transformers 新版、EffectErase 要 diffusers 0.30-0.31 + transformers<5，
装在一起会打架）。

六个 worker 全部遵守同一份契约：

| 约定 | 内容 |
|---|---|
| 入参 | 命令行参数，含 `--out <path>` |
| 出参 | 把结果 JSON 写进 `--out` 指向的文件 |
| 兜底 | 同时把 `<NAME>_RESULT {json}` 打到 stdout 一行 |
| 解析 | wrapper 先读文件，读不到再从 stdout 找 marker 行，都失败才报错并附 stdout 末尾 2000 字 |

现有的八个 marker：`QWEN3_OMNI_RESULT`、`SAM3_RESULT`、`SAM_AUDIO_RESULT`、
`EFFECTERASE_RESULT`、`LTX_ENHANCE_RESULT`、`IB_SELECT_RESULT`、`CLAP_SELECT_RESULT`、
`ACOUSTIC_DESC_RESULT`。

`ACOUSTIC_DESC_RESULT`（`acoustic_desc_worker.py`，由 `ACOUSTIC_DESC=clap` 启用，默认 off）在抽取
节点之后给目标一个**声学描述**（原始音频的 CLAP 零样本标签，如 `footsteps`），`AUDIO_TEXT=acoustic`
时音频侧三处（分离 prompt / best-of 选优 / 检查）改用它，视觉侧不变。为什么：已交付人物类样本
90% 无人声，在响的是同一个人的动作声，用 `man` 提示音频侧拿不到信号（strategy-lab S55/S56/S59）。

其中 `IB_SELECT_RESULT` 与 `CLAP_SELECT_RESULT` 是同一个阶段（best-of 选优）的两个可换
选优器，由 `BESTOF_SELECTOR=ib|clap` 切换，默认 `ib`（历史行为逐位不变）。
为什么有第二个：ImageBind 键与移除质量在样本内秩相关 +0.02，换键无增益；CLAP 的
`s_mix − s_res` 有（avgraph-strategy-lab S52/S53）。增益是回测上限，开之前要在新生成上 A/B。

**所以换模型 = 写一个遵守这份契约的新 worker**，加上让 wrapper 指向它
（`python_bin` 与 `worker` 都是 `__init__` 参数）。不需要动图、路由或任何节点。

## 三档可换性——说清楚哪档现在真的成立

| 档次 | 含义 | 现状 |
|---|---|---|
| **换 checkpoint** | 同架构、换权重 | **全部支持**，走 env 或参数（如 `SAM3_MODEL_DIR`、`QWEN3_OMNI_PATH`） |
| **换同族架构** | 如 Qwen3-Omni → Qwen2.5-Omni | 只有 captioner 支持（`--model_class` / `--processor_class`） |
| **换跨族模型** | 如 SAM3 → 别的分割器 | **都要写新 worker**，这是设计如此，不是缺陷 |

### captioner 的可换性到哪为止

2026-08-30 我把 `qwen3_omni_worker.py` 的模型类改成了参数注入，但**那只解开了
一处耦合**。`caption_video()` 里还有两处 Qwen-Omni 家族特有的东西：

- `process_mm_info`（来自 `qwen_omni_utils`）
- `generate(..., thinker_return_dict_in_generate=True, return_audio=False)`
  ——thinker/talker 双头架构特有的参数

Qwen2.5-Omni 与 Qwen3-Omni 同属该家族、同为 thinker/talker 结构，所以**大概率能跑**，
但没有在真机上验证过（本地无 transformers，不猜类名也不猜兼容性）。
换到非 Qwen 的 omni 模型则必须写新 worker。

**不要因为有了 `--model_class` 就以为 captioner 可以随便换。**

## 写一个新 worker 需要满足什么

1. 接受 `--out <path>`，把结果 JSON 写进去
2. 同时 `sys.stdout.write(MARKER + json.dumps(result) + "\n")` 并 flush
3. 结果 JSON 的字段要与对应 wrapper 的解析代码对齐
   （如分割 worker 要给 `first_frame_ratio` / `passed` / `mask_path` / `n_instances`）
4. 住在自己的 conda env 里，通过 wrapper 的 `python_bin` 指过去

## 换了之后怎么知道是变好还是变坏

这是换模型的全部意义所在，所以单独说：

每条落盘记录都带 `config` 块（2026-08-30 加），记录四道闸的阈值、首帧闸的实际值
与预期值、`gate_relaxed` 标记、`gates_measuring`（哪几道闸真的在测量）、
以及 captioner 与 extraction 的模型身份。

**比较两次运行前，先核对两者的 `config` 块是否只在你要换的那一项上不同。**
不这么做的代价有实测：8,907 条历史运行没记录闸档，h₀ = 21.84% 产生自首帧闸 0.05，
而代码预期值是 0.80，按预期值回算交付量从 1,945 条掉到 45 条——同一个指标
在两档之间差 40 倍，而两者在日志里长得一模一样。

还要注意 `gates_measuring` 目前是 `{mask: true, visual: true, audio: false,
cross_modal: false}`：**换音频模型时，音频闸给不出任何信号**（分数是硬编码常数
0.90，1,946 条里 1,945 条恰为该值）。在音频闸接线之前，音频侧的换模型对比
只能靠人工听或另测。
