# asr-module 设计文档

## 1. 概述

`asr_debug.py` 是一个本地离线 ASR（自动语音识别）CLI 调试工具，专为具身智能机器人项目的语音指令调试设计。

**核心目标：**
- 低延迟（从说完到出字 < 500ms）
- 全离线（不依赖网络 API）
- 交互简单（空格触发，VAD 自动停止）
- macOS 开箱即用

---

## 2. 技术选型

### 2.1 ASR 模型：SenseVoice-Small

| 特性 | 说明 |
|------|------|
| 来源 | 阿里巴巴 / FunASR 团队，2024年发布 |
| 参数量 | ~250MB |
| 推理速度 | 比 Whisper-large-v3 快 **15倍** |
| 语言支持 | 中文、英文、日文、韩文、粤语 |
| 额外能力 | 情绪识别（NEUTRAL/HAPPY/SAD...）、音频事件检测 |
| 部署方式 | `funasr` + `modelscope` 自动下载 |

SenseVoice-Small 在中文短句识别场景下的 WER（词错率）显著低于同量级的 Whisper，且推理速度极快，非常适合机器人实时指令场景。

### 2.2 VAD 模型：fsmn-vad

| 特性 | 说明 |
|------|------|
| 来源 | 阿里巴巴 / FunASR，工业级 |
| 模型类型 | FSMN（Feedforward Sequential Memory Network） |
| 处理模式 | 支持流式（在线）处理 |
| 输出格式 | `[[beg_ms, end_ms]]`，`-1` 表示该端点尚未发生 |
| 用途 | 精确检测语音起止时刻，避免无效 ASR 调用 |

fsmn-vad 相比简单能量阈值 VAD 的优势：
- 对背景噪声鲁棒
- 能准确识别短句（不会因为中间短暂停顿误判为结束）
- 延迟约 200~400ms（等待一段静音确认结束）

---

## 3. 架构设计

### 3.1 数据流

```
麦克风（16kHz float32）
    │
    ▼ 每 200ms 一块（3200 samples）
sounddevice.InputStream
    │
    ▼ audio_queue（线程安全队列）
主处理循环
    ├─► fsmn-vad（流式，带 cache 保持状态）
    │       │
    │       ├─ 检测到 beg ≥ 0 → 打印 [VAD_START]，开始缓存音频
    │       └─ 检测到 end ≥ 0 → 打印 [VAD_END]，停止录音
    │
    └─► speech_buffer（只缓存 VAD_START 之后的音频）
            │
            ▼ 语音结束后
        SenseVoiceSmall（批量推理）
            │
            ▼
        打印识别结果（含耗时）
```

### 3.2 线程模型

```
主线程
  ├── sounddevice 回调线程（系统级，填充 audio_queue）
  └── 主循环（消费 audio_queue，运行 VAD + ASR）
```

设计为单一主循环处理 VAD 和 ASR，避免多线程竞争，逻辑简单清晰。

### 3.3 流式 VAD 状态机

```
[等待语音]
    │
    │ fsmn-vad 输出 beg ≥ 0
    ▼
[语音进行中]  ← 打印 [VAD_START]，缓存音频
    │
    │ fsmn-vad 输出 end ≥ 0
    ▼
[语音结束]  ← 打印 [VAD_END]，停止采集
    │
    ▼
[ASR 推理]  ← SenseVoiceSmall 识别 speech_buffer
    │
    ▼
[打印结果]  → 回到主循环等待下次空格
```

---

## 4. 关键实现细节

### 4.1 fsmn-vad 流式调用

```python
vad_cache = {}  # 在每次录音中保持，跨 chunk 维持状态

res = vad_model.generate(
    input=chunk_np,      # 当前 200ms 音频块
    cache=vad_cache,     # 内部状态（不要清空）
    is_final=False,      # 最后一块设为 True
    chunk_size=200,      # 与 CHUNK_MS 保持一致
)
```

`cache` 字典由 fsmn-vad 内部维护，存储 FSMN 的历史状态，**不要在 chunk 间清空**，只在每次新的录音会话开始时重置（`vad_cache = {}`）。

### 4.2 音频只缓存语音部分

```python
# 只在 speech_started=True 且 speech_ended=False 时缓存
if speech_started and not speech_ended:
    speech_buffer.append(chunk_np)
```

这样 ASR 只处理有效语音，不包含前置静音，减少推理时间。

### 4.3 SenseVoice 情绪标签过滤

SenseVoice 输出中可能包含情绪标签，如 `<|NEUTRAL|>向前走五十厘米`，需要过滤：

```python
import re
text = re.sub(r"<\|[A-Z_]+\|>", "", raw).strip()
```

后续集成到机器人大脑时，情绪标签可以保留作为辅助信息使用。

### 4.4 macOS 键盘无回显读取

使用 `termios` + `tty` 实现按键即响应（无需回车）：

```python
def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch
```

`tty.setraw()` 将终端切换到原始模式，读取单字符后立即恢复，不依赖 `pynput`（无需辅助功能权限）。

---

## 5. 性能参考

在 2018 MacBook Pro（Intel Core i7）上的实测数据（大致参考）：

| 阶段 | 耗时 |
|------|------|
| VAD 检测（per chunk） | < 10ms |
| fsmn-vad 语音结束检测延迟 | 200~600ms（取决于句末静音长度） |
| SenseVoice 识别（5s 语音） | 300~800ms |
| 端到端总延迟（从停止说话到出字） | **500ms ~ 1.5s** |

在 Apple Silicon（M2+）上所有阶段均会快 3~5 倍。

---

## 6. 后续扩展计划

### 6.1 接入 LLM（指令理解）

```
识别结果 → Qwen2.5-1.5b（Ollama）→ skill_call → ESP32
```

### 6.2 流式识别（边说边出字）

使用 FunASR 的 `Paraformer-zh-streaming` 模型，支持实时逐字输出，配合 VAD 做端点检测。

### 6.3 唤醒词支持

集成 `sherpa-onnx` 的关键词检测，实现「嘿机器人」自动激活，无需按空格。

---

## 7. 依赖版本说明

| 库 | 最低版本 | 说明 |
|----|---------|------|
| funasr | 1.1.0 | 修复了 fsmn-vad 流式 cache 的多个 bug |
| modelscope | 1.9.0 | 支持 iic/ 命名空间的模型下载 |
| sounddevice | 0.4.6 | 修复了 macOS 14+ 的音频采集兼容性 |
| numpy | 1.24.0 | dtype float32 处理要求 |
