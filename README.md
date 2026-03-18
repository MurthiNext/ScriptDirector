<h1><p align='center' >Script Director</p></h1>
<div align=center><img src="https://img.shields.io/github/v/release/MurthiNext/ScriptDirector"/>   <img src="https://img.shields.io/github/license/MurthiNext/ScriptDirector"/>   <img src="https://img.shields.io/github/stars/MurthiNext/ScriptDirector"/></div>

### &emsp;&emsp;Script Director 是一个将音频文件与台本（文本）自动对齐，生成带时间戳字幕（SRT/LRC）的工具。它利用 **Faster Whisper** 进行语音识别，并通过 **Needleman-Wunsch** 风格的动态规划算法将识别结果与台本句子精确匹配，即使识别结果与台本不完全一致也能智能插值，确保每一句台本都有准确的时间码。

## 特性

- 🎙️ 基于 Faster Whisper 的高质量语音识别
- 📄 支持日语（`ja`）、中文（`zh`）、英文（`en`）等多语言台本分割（使用 `pysbd`）
- 🔗 智能句子对齐：采用 Needleman-Wunsch 风格算法，处理插入、删除和替换
- ⏱️ 对未匹配的句子进行线性插值，确保字幕完整
- 📝 输出格式自动识别：根据输出文件后缀生成 SRT 或 LRC
- ⚙️ 命令行界面（CLI）和配置文件支持，方便重复使用
- 🤖 这篇README也是由DeepSeek V3.2写的，拜托，正经人谁会写文档嘛

## 安装

### 依赖
- Python 3.8+
- [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)（需预下载 CTranslate2 格式模型）
- 其他 Python 包：`pysbd`, `rapidfuzz`, `click`

### 安装步骤
1. 克隆或下载本项目。
2. 安装依赖：
   ```bash
   pip install faster-whisper pysbd rapidfuzz click
   ```
3. 下载 Faster Whisper 模型（例如 [faster-whisper-large-v3](https://huggingface.co/guillaumekln/faster-whisper-large-v3)）并解压到本地目录。

## 使用方法

Script Director 提供命令行工具 `cli.py`，支持配置文件 `config.ini` 存储常用参数（模型路径、语言、设备等）。

### 首次运行配置
首次运行 `cli.py` 时，程序会引导你创建配置文件：
```bash
python cli.py
```
按照提示输入：
- **Faster Whisper 本地模型路径**：模型文件夹的路径（如 `./faster-whisper-large-v3`）
- **台本与音频所使用的语言代码**：例如 `ja`（日语）、`zh`（中文）、`en`（英文）
- **设备类型**：`cuda` 或 `cpu`
- **计算类型**：`float16`（GPU）、`int8`（CPU）等

配置完成后会在当前目录生成 `config.ini`。

### 基本命令
```bash
python cli.py --help
```
- `--help`：获取命令行帮助。
```bash
python cli.py -i "音频文件路径,台本文件路径" -o 输出文件路径
```
- `-i, --input`：音频文件和台本文件路径，用英文逗号分隔（例如 `audio.wav,script.txt`）
- `-o, --output`：输出文件路径，扩展名决定格式：`.srt` 生成 SRT 字幕，`.lrc` 生成 LRC 歌词（默认 `output.lrc`）

#### 示例
```bash
python cli.py -i "meeting.wav,transcript.txt" -o subtitles.srt
```
程序将：
1. 使用配置的模型对 `meeting.wav` 进行语音识别。
2. 读取台本 `transcript.txt` 并分割为句子。
3. 对齐识别结果与台本，生成时间戳。
4. 保存为 SRT 字幕文件 `subtitles.srt`。

### 配置说明
`config.ini` 文件内容示例：
```ini
[common]
model = D:/models/faster-whisper-large-v3
lang = ja
device = cuda
compute = float16
```
- `model`：Faster Whisper 模型文件夹路径
- `lang`：语言代码（支持 `ja`, `zh`, `en`, `ko`, `fr`, `de` 等）
- `device`：计算设备 `cuda` 或 `cpu`
- `compute`：计算精度，常用 `float16`（GPU）或 `int8`（CPU）

如需修改配置，可直接编辑 `config.ini` 或删除后重新运行 `cli.py`。

## 项目结构
- `director.py`：核心模块，包含语音识别、句子对齐、时间戳映射、字幕保存等功能。
- `cli.py`：命令行入口，处理参数、配置文件并调用 `director.direct_it`。
- `app.py`：图形化界面入口，正在开发中……

## 注意事项
- 音频格式支持取决于 Faster Whisper（常见格式如 `wav`, `mp3`, `m4a` 等）。
- 台本文件需为 UTF-8 编码的纯文本。
- 如果 Whisper 识别结果与台本差异较大，可尝试调整 `align_sentence_lists` 中的 `gap_penalty` 参数（当前硬编码为 `-10`）。

## 许可证
* 本项目采用 **MIT 许可证**。详情请参阅 [LICENSE](LICENSE) 文件。

---

**Happy Subtitling!** 🎬
