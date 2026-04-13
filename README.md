# asr-module

> 本地离线 ASR 调试工具，基于 **SenseVoice-Small**（阿里 FunASR）+ **fsmn-vad** 实现。
>
> 按空格键开始录音，VAD 自动检测语音起止，识别完毕打印结果，循环等待下次输入。

---

## 快速开始

### 1. 环境准备

> ⚠️ **要求 Python 3.9 ~ 3.12**，macOS / Linux
>
> PyTorch 官方目前**不支持 Python 3.13+**。
> 如果系统默认 `python3` 是 3.13/3.14，请用 `python3.12` 显式指定版本创建 venv：

```bash
# macOS 先安装 PortAudio（sounddevice 依赖）
brew install portaudio

# 用 Python 3.12 创建虚拟环境（避免 Python 3.13+ 与 torch 不兼容）
python3.12 -m venv .venv
source .venv/bin/activate

# 安装全部依赖（含 torch）
pip install -r requirements.txt
```

> 💡 如果没有 python3.12，可通过 Homebrew 安装：
> ```bash
> brew install python@3.12
> ```

### 2. 运行

```bash
python asr_debug.py
```

首次运行会自动下载模型（`~/.cache/modelscope/`），大约需要几分钟，后续秒启。

---

## 终端输出示例

```
────────────────────────────────────────────────────
  🎤  ASR 调试工具  (SenseVoice + fsmn-vad)
────────────────────────────────────────────────────

✅ 模型加载完成

  按 [空格] 开始录音，Ctrl+C 退出

⏸  等待空格...

▶  开始录音...  （请讲话，静音后自动停止）
[VAD_START]
[VAD_END]
🔍  识别中...
📝  结果：向前走五十厘米  (312ms)

⏸  等待空格...
```

---

## 操作说明

| 按键 | 行为 |
|------|------|
| `空格` | 开始录音（VAD 自动检测语音，无需手动停止） |
| `Ctrl+C` | 退出程序 |

---

## 输出标记含义

| 标记 | 含义 |
|------|------|
| `[VAD_START]` | fsmn-vad 检测到语音开始 |
| `[VAD_END]` | fsmn-vad 检测到语音结束（静音超过阈值） |
| `📝  结果：...` | SenseVoice-Small 识别结果（含识别耗时） |

---

## 配置参数

在 `asr_debug.py` 顶部可调整以下参数：

```python
CHUNK_MS    = 200   # VAD 处理块大小（毫秒），越小延迟越低但 CPU 占用越高
MAX_RECORD_S = 30   # 单次最长录音秒数（超时自动停止）
```

---

## 依赖说明

| 库 | 用途 |
|----|------|
| `funasr` | FunASR 推理框架，提供 SenseVoice + fsmn-vad |
| `sounddevice` | 麦克风实时采集（16kHz float32） |
| `numpy` | 音频数组拼接处理 |
| `modelscope` | 模型自动下载 |

---

## 目录结构

```
asr-module/
├── asr_debug.py      ← 主程序
├── requirements.txt  ← 依赖清单
├── README.md         ← 本文件
└── doc/
    └── design.md     ← 架构设计文档
```

---

## 后续扩展

- 接入 Ollama LLM，将识别结果发送给大模型理解意图
- 通过 WebSocket 将识别结果发送到 ESP32 底盘（vehicle-module）
- 支持流式识别（边说边出字）
