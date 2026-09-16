# 提取节点 prompt 的根因与修法

## 根因（在哪一行）

`models/object_extraction_model.py`，`SYSTEM_PROMPT`：

```
"An object qualifies only if it has a clear physical mechanism for sound production "
"(e.g. vibrating strings, flowing water, a running engine, a speaking person, "
"flapping wings, crackling fire). "
```

**`a running engine` 被明写为合格示例。** GPT-4o-mini 输出 `car engine` 不是它出错——
它精确地照做了。整条 prompt 定义了「发声」，**从未要求输出必须在视觉上可分割**，
而下游 SAM3 需要的正是后者。

两个节点各自都对，**接口处的假设没人写下来**。

## 后果（实测，n=8907）

| 目标 | 样本 | 通过率 |
|---|---|---|
| `car engine` | 665 | **0.2%** |
| `vehicle engine` | 140 | 0% |
| `truck engine` | 84 | 0% |
| `motorcycle engine` | 76 | 0% |
| `bus engine` | 61 | 0% |
| `footsteps` | 40 | 0% |
| 全管线基线 | 8907 | 21.84% |

自动契约检测器（`avgraph-strategy-lab/experiments/contract_check.py`）扫出
**32 个这样的取值，覆盖 2902 条样本、只产出 41 条、浪费 2861 条**。

## 为什么不能在下游修

运行时把 `car engine` 换成 `car` 会**破坏音视频硬一致性约束**：
音频侧 SAM-Audio 的提示词仍是 `car engine`（去掉引擎声），
视觉侧却擦掉整辆车——两侧移除的不是同一个东西。
这条约束是 correctness，不是偏好，零自由度。

（`man's voice → person` 更明显：人除了说话还有脚步声、衣物摩擦声。
擦掉整个人却只去掉说话声，剩下的脚步声就成了「看不见的人在走路」，
而音频 gate 是空操作，检测不出来。）

## 修法：改 prompt，让它一次产出合格的东西

```python
SYSTEM_PROMPT = (
    "You are an audio-visual scene analyst. Given a video caption, identify the "
    "objects that are BOTH actively producing sound AND visually segmentable in "
    "the frame.\n"
    "\n"
    "An object qualifies only if BOTH hold:\n"
    "  (a) it has a clear physical mechanism for sound production, and\n"
    "  (b) it is a visible, boundable entity that could be outlined in the image.\n"
    "\n"
    "Critically: if the sound-producing mechanism is hidden inside or is a part of "
    "a larger object (an engine inside a car, a speaker inside a phone), name the "
    "VISIBLE WHOLE, not the hidden part — output 'car', not 'car engine'.\n"
    "If the sound is an event or action rather than an object (footsteps, speech, "
    "a horn honking), name the visible entity producing it — output 'person', not "
    "'footsteps'.\n"
    "Exclude sounds with no boundable visual source (wind, ambient noise, "
    "background music, rain) — these have no object to segment.\n"
    "Exclude static background objects that do not themselves emit sound "
    "(buildings, walls, trees, roads).\n"
    "\n"
    "Output ONLY a comma-separated list of the qualifying objects, most prominent "
    "first. If nothing qualifies, output: none."
)
```

三处关键改动：

1. **(b) 视觉可分割成为硬条件**——这是原 prompt 缺失的那一半契约。
2. **隐藏部件 → 可见整体**，并直接用 `car engine` 当反例（原 prompt 用它当正例）。
3. **声音事件 → 发声实体**，`footsteps` → `person`。

一致性由此自然保持：**从头到尾只有一个词**，音频侧和视觉侧用同一个，
不存在「换目标」这回事。

## 待验证（这才是实验该测的）

**不是**「新 prompt 能不能救回 engine 类」——那几乎必然（它们现在是 0%）。
**而是会不会伤到现在能过的那些**：`man` 48%、`woman` 55%、`dog` 30%、`car` 34%。
新 prompt 更严格，可能把一些本来能过的样本判成 `none`。

所以 A/B 必须**同时抽失败样本和成功样本**，比较：
- 失败类：新 prompt 是否产出可分割的词（提升）
- 成功类：新 prompt 是否仍产出同样的词（无回归）

**阻塞**：历史 caption **一条都没存**（`data/work/captions` 为空），
8907 × 10 分钟 ≈ 1484 GPU-小时的 caption 全部丢失。要 A/B 必须重新生成，
而 Qwen3-Omni 是 66GB MoE，空闲卡 46GB 放不下，需 CPU offload。

## 顺带发现的两个日志缺口

- `sounding_objects` 列表（提取节点的完整输出）**没有落盘**，只存了最终 `object_name`。
  所以无法回答「同一条片子里是否本来就有可分割的备选发声物体」——
  而那可能是比改 prompt 更好的修法。
- caption 没落盘。这两项都应该补进 `utils.py` 的写盘逻辑。
