#!/usr/bin/env python3
"""
asr_debug.py — CLI ASR 调试工具

用法：
    python asr_debug.py                # 使用系统默认麦克风
    python asr_debug.py --list         # 列出所有输入设备
    python asr_debug.py --device 3     # 指定设备 ID（用 --list 查看）

操作：
    按 [空格]   → 开始录音，VAD 自动检测语音起止，识别后打印结果
    按 Ctrl+C  → 退出
"""

import sys
import queue
import threading
import termios
import tty
import time
import argparse
from collections import deque

import numpy as np
import sounddevice as sd

# ── ANSI 颜色 ──────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
CYAN    = "\033[96m"
GRAY    = "\033[90m"
RED     = "\033[91m"

# ── 音频参数 ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16_000
CHUNK_MS    = 200
CHUNK_SIZE  = int(SAMPLE_RATE * CHUNK_MS / 1000)   # 3200 samples

# ── 超时设置 ───────────────────────────────────────────────────────────────────
MAX_RECORD_S = 30

# ── 预缓冲（pre-roll）：VAD_START 前保留多少毫秒的音频 ────────────────────────
# 解决"丢开头字"问题：VAD 检测到语音时，把之前这段音频也带上
PRE_ROLL_MS = 600   # 600ms ≈ 3 个 CHUNK；可按需调大到 800ms


# ──────────────────────────────────────────────────────────────────────────────
# 设备管理
# ──────────────────────────────────────────────────────────────────────────────

def list_input_devices():
    """打印所有可用输入设备。"""
    devs = sd.query_devices()
    default_in = sd.default.device[0]
    print(f"\n{BOLD}{CYAN}可用输入设备：{RESET}")
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            tag = f"{GREEN} ◀ DEFAULT{RESET}" if i == default_in else ""
            print(f"  [{i:2d}] {d['name']}{tag}")
    print()


def pick_device(device_id=None):
    """
    选择输入设备。
    - device_id=None  → 使用系统默认
    - device_id=int   → 使用指定 ID
    - 自动提示：若默认设备名包含 'Built-in'，提示用户可用 --device 切换
    """
    devs = sd.query_devices()
    default_in = sd.default.device[0]

    if device_id is not None:
        dev = devs[device_id]
        print(f"{CYAN}🎙  使用设备 [{device_id}]: {dev['name']}{RESET}")
        return device_id

    # 默认设备
    dev = devs[default_in]
    name = dev['name']
    print(f"{CYAN}🎙  使用默认设备 [{default_in}]: {name}{RESET}")

    # 友好提示：若默认是内置麦克风，提醒用户检查
    if 'Built-in' in name or 'MacBook' in name:
        print(f"{YELLOW}   ⚠️  检测到内置麦克风。如需使用耳机/外接麦克风，请运行：{RESET}")
        print(f"{GRAY}      python asr_debug.py --list           # 查看设备列表{RESET}")
        print(f"{GRAY}      python asr_debug.py --device <ID>   # 指定设备{RESET}")
    return None   # None = sounddevice 使用系统默认


# ──────────────────────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────────────────────

def load_models():
    print(f"{YELLOW}⏳ 正在加载模型（首次运行需下载，约数分钟）...{RESET}", flush=True)
    from funasr import AutoModel

    vad = AutoModel(
        model="fsmn-vad",
        model_revision="v2.0.4",
        disable_log=True,
        disable_pbar=True,
    )
    asr = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        disable_log=True,
        disable_pbar=True,
    )
    print(f"{GREEN}✅ 模型加载完成{RESET}\n", flush=True)
    return vad, asr


# ──────────────────────────────────────────────────────────────────────────────
# 键盘输入
# ──────────────────────────────────────────────────────────────────────────────

def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ──────────────────────────────────────────────────────────────────────────────
# 音量电平显示
# ──────────────────────────────────────────────────────────────────────────────

def _level_bar(rms: float, width: int = 30) -> str:
    """将 RMS 值（0~1）转为彩色进度条。"""
    db = 20 * np.log10(rms + 1e-9)   # dBFS，-90 ~ 0
    db_norm = max(0.0, min(1.0, (db + 60) / 60))   # -60dBFS ~ 0dBFS → 0~1
    filled = int(db_norm * width)
    color = GREEN if db_norm < 0.7 else YELLOW if db_norm < 0.9 else RED
    bar = color + "█" * filled + RESET + GRAY + "░" * (width - filled) + RESET
    return f"[{bar}] {db:5.1f}dB"


# ──────────────────────────────────────────────────────────────────────────────
# 录音 + VAD + ASR
# ──────────────────────────────────────────────────────────────────────────────

def record_and_recognize(vad_model, asr_model, device=None):
    audio_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def _audio_callback(indata, frames, time_info, status):
        if not stop_event.is_set():
            audio_queue.put(indata.copy())

    print(f"\n{CYAN}▶  开始录音...  {GRAY}（请讲话，静音后自动停止）{RESET}", flush=True)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        device=device,          # None = 系统默认；int = 指定设备
        callback=_audio_callback,
    )

    vad_cache      = {}
    speech_buffer  = []
    speech_started = False
    speech_ended   = False
    elapsed_ms     = 0

    # pre-roll：滚动保存语音前的音频，VAD_START 时一起送给 ASR
    pre_roll_size = PRE_ROLL_MS // CHUNK_MS
    pre_roll: deque = deque(maxlen=pre_roll_size)

    with stream:
        while not speech_ended:
            if elapsed_ms > MAX_RECORD_S * 1000:
                print(f"\n{YELLOW}⚠️  超过最长录音时间（{MAX_RECORD_S}s），强制停止{RESET}", flush=True)
                break

            try:
                chunk = audio_queue.get(timeout=3.0)
            except queue.Empty:
                print(f"\n{GRAY}（3 秒无音频输入，停止录音）{RESET}")
                break

            chunk_np = chunk.flatten().astype(np.float32)
            elapsed_ms += CHUNK_MS

            # ── 实时音量电平 ─────────────────────────────────────────────────
            rms = float(np.sqrt(np.mean(chunk_np ** 2)))
            bar = _level_bar(rms)
            status_tag = f"{GREEN}[语音]{RESET}" if speech_started else f"{GRAY}[静音]{RESET}"
            print(f"\r  {status_tag} {bar}  {elapsed_ms/1000:.1f}s", end="", flush=True)

            # ── fsmn-vad ─────────────────────────────────────────────────────
            res = vad_model.generate(
                input=chunk_np,
                cache=vad_cache,
                is_final=False,
                chunk_size=CHUNK_MS,
            )
            vad_events = res[0].get("value", []) if res else []

            for seg in vad_events:
                beg, end = seg[0], seg[1]
                if not speech_started and beg >= 0:
                    speech_started = True
                    # 把 pre-roll 音频拼到 speech_buffer 开头，补回丢失的开头
                    speech_buffer.extend(list(pre_roll))
                    print(f"\n{GREEN}[VAD_START]{RESET} {GRAY}(pre-roll {len(pre_roll)*CHUNK_MS}ms){RESET}", flush=True)
                if speech_started and end >= 0:
                    speech_ended = True
                    print(f"\n{GREEN}[VAD_END]{RESET}", flush=True)
                    stop_event.set()
                    break

            if speech_started and not speech_ended:
                speech_buffer.append(chunk_np)
            else:
                # 语音尚未开始，把 chunk 存入滚动预缓冲
                pre_roll.append(chunk_np)

        # flush VAD
        if speech_started and not speech_ended:
            final_np = np.zeros(CHUNK_SIZE, dtype=np.float32)
            res = vad_model.generate(
                input=final_np, cache=vad_cache,
                is_final=True, chunk_size=CHUNK_MS,
            )
            vad_events = res[0].get("value", []) if res else []
            for seg in vad_events:
                if seg[1] >= 0 and not speech_ended:
                    speech_ended = True
                    print(f"\n{GREEN}[VAD_END]{RESET}", flush=True)

    print()  # 电平行换行

    if not speech_buffer:
        print(f"{GRAY}  （未检测到语音）{RESET}")
        print(f"{YELLOW}  💡 提示：若音量电平始终很低（< -50dB），说明录音设备不对，请用 --list 查看并用 --device 指定正确的麦克风。{RESET}")
        return

    print(f"{YELLOW}🔍  识别中...{RESET}", flush=True)
    full_audio = np.concatenate(speech_buffer)

    t0 = time.time()
    result = asr_model.generate(
        input=full_audio,
        cache={},
        language="zh",
        use_itn=True,
    )
    elapsed = (time.time() - t0) * 1000

    text = ""
    if result and result[0].get("text"):
        import re
        raw = result[0]["text"]
        text = re.sub(r"<\|[A-Z_]+\|>", "", raw).strip()

    if text:
        print(f"{BOLD}{GREEN}📝  结果：{text}{RESET}  {GRAY}({elapsed:.0f}ms){RESET}", flush=True)
    else:
        print(f"{GRAY}  （未识别到文字）{RESET}")


# ──────────────────────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CLI ASR 调试工具 (SenseVoice + fsmn-vad)")
    parser.add_argument("--list",   action="store_true", help="列出所有可用输入设备后退出")
    parser.add_argument("--device", type=int, default=None, metavar="ID", help="指定输入设备 ID（用 --list 查看）")
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        sys.exit(0)

    print(f"\n{BOLD}{CYAN}{'─' * 52}{RESET}")
    print(f"{BOLD}  🎤  ASR 调试工具  (SenseVoice + fsmn-vad){RESET}")
    print(f"{BOLD}{CYAN}{'─' * 52}{RESET}\n")

    device = pick_device(args.device)
    print()

    vad_model, asr_model = load_models()
    print(f"  按 {BOLD}{CYAN}[空格]{RESET} 开始录音，{GRAY}Ctrl+C 退出{RESET}\n")

    try:
        while True:
            print(f"{GRAY}⏸  等待空格...{RESET}", flush=True)
            ch = getch()
            if ch == "\x03":
                print(f"\n{GRAY}👋 退出{RESET}")
                sys.exit(0)
            if ch == " ":
                record_and_recognize(vad_model, asr_model, device=device)
                print()
    except KeyboardInterrupt:
        print(f"\n{GRAY}👋 退出{RESET}")


if __name__ == "__main__":
    main()
