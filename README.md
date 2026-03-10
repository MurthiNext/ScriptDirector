# 台本转字幕工具 (Script to Subtitles)

本工具利用 **Faster Whisper** 自动语音识别（ASR）和文本对齐技术，将给定的台本（文本文件）与音频文件对齐，生成带时间戳的字幕文件。支持输出 **SRT** 或 **LRC** 格式，自动根据输出文件后缀名判断。

- 本README由DeepSeek V3.2提供支持，写README什么的最讨厌了！

---

## 功能特点

- **多格式支持**：自动识别输出文件后缀 `.srt` 或 `.lrc`，生成对应的字幕/歌词文件。
- **精确对齐**：使用动态规划（Needleman-Wunsch）算法将台本句子与 Whisper 识别的句子片段对齐，有效处理插入、删除和替换。
- **日文优化**：默认支持日语（`language='ja'`），通过 `pysbd` 进行准确的句子边界检测。
- **本地模型加载**：直接使用本地 CTranslate2 格式的 Faster Whisper 模型，无需在线下载。
- **GPU加速**：支持 CUDA 加速（默认 `device='cuda'`），可快速处理长音频。

---

## 安装依赖

确保 Python 环境为 3.8 或更高版本，然后安装以下依赖：

```bash
pip install faster-whisper pysbd rapidfuzz
```

**可选**：如需 GPU 加速，请确保已安装 CUDA 和 cuDNN，并安装对应版本的 `faster-whisper`（通常会自动匹配）。

---

## 使用方法

### 1. 准备文件
- **音频文件**：支持常见格式如 `.wav`, `.mp3` 等。
- **台本文件**：纯文本文件，包含音频中出现的全部文本，编码为 UTF-8。
- **本地模型**：下载 CTranslate2 格式的 Faster Whisper 模型（例如 [faster-whisper-large-v3](https://huggingface.co/guillaumekln/faster-whisper-large-v3)），解压到本地文件夹。

### 2. 修改脚本中的路径
直接修改 `if __name__ == "__main__":` 部分中的参数，或通过函数参数传递。

```python
if __name__ == "__main__":
    main(
        audio_path="audio.wav",                # 音频文件路径
        script_path="script.txt",               # 台本文件路径
        output_path="output.lrc",                # 输出文件路径（.srt 或 .lrc）
        local_model_path="./faster-whisper-large-v3-turbo",  # 本地模型文件夹路径
        language='ja',                           # 语言代码（'ja'日语, 'zh'中文, 'en'英文等）
        device='cuda',                           # 计算设备 'cuda' 或 'cpu'
        compute_type='float16'                    # 计算类型（如 'float16', 'int8_float16'）
    )
```

### 3. 运行脚本
```bash
python your_script_name.py
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio_path` | str | 必填 | 输入音频文件路径 |
| `script_path` | str | 必填 | 台本文件路径（UTF-8编码） |
| `output_path` | str | 必填 | 输出文件路径，后缀决定格式（`.srt` 或 `.lrc`） |
| `local_model_path` | str | 必填 | 本地 CTranslate2 格式的 Whisper 模型文件夹 |
| `language` | str | `'ja'` | 音频语言代码（ISO 639-1），如 `'ja'` 日语，`'zh'` 中文 |
| `device` | str | `'cuda'` | 计算设备，可选 `'cuda'` 或 `'cpu'` |
| `compute_type` | str | `'float16'` | 模型计算精度，常用值 `'float16'`（GPU）、`'int8_float16'`、`'int8'` |

---

## 输出说明

### SRT 格式（示例）
```
1
00:00:01,000 --> 00:00:04,500
これは最初の文です。

2
00:00:05,200 --> 00:00:08,300
これは二番目の文です。
```

### LRC 格式（示例）
```
[00:01.00] これは最初の文です。
[00:05.20] これは二番目の文です。
```

---

## 注意事项

1. **语言支持**：`pysbd` 的句子分割功能依赖指定语言，确保 `language` 参数与音频及台本的语言一致。  
2. **模型路径**：`local_model_path` 必须指向包含 `model.bin`、`config.json`、`tokenizer.json` 等文件的 CTranslate2 模型文件夹。  
3. **音频时长**：Faster Whisper 会自动处理长音频，但内存占用与音频时长有关，建议在 GPU 上运行以加快速度。  
4. **对齐效果**：如果台本与音频内容差异较大（如 Whisper 识别错误较多），可能需要调整 `gap_penalty` 或相似度计算方法。  
5. **错误处理**：时间格式化函数已添加异常捕获，避免因无效时间戳导致程序崩溃。

---

## 常见问题

### Q: 输出时间戳为 `00:00:00,000` 或 `[00:00.00]`？
A: 这通常表示从 Whisper 获取的时间戳无效（例如 None 或字符串）。请检查音频是否能正常转录，或更新 Faster Whisper 版本。

### Q: 如何切换为英文或中文？
A: 修改 `language` 参数为 `'en'`（英文）或 `'zh'`（中文），并确保 `pysbd` 支持该语言（中文需安装 `pysbd` 的额外语言包，或使用其他分割器）。

### Q: 没有 GPU，如何使用 CPU 运行？
A: 将 `device` 参数设为 `'cpu'`，`compute_type` 可设为 `'int8'` 以减少内存占用。

### Q: 对齐后的字幕数量少于台本句子数？
A: 某些句子可能因匹配分数过低被跳过。可调整 `align_sentence_lists` 中的 `gap_penalty` 或相似度阈值（硬编码在 DP 得分计算中）以提高容忍度。

---

## 许可证

本项目使用MIT License。

---

如果有任何问题或建议，欢迎提交 Issue 或改进代码！
