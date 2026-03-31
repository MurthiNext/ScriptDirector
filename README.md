<h1><p align='center' >Script Director ROCm-Version</p></h1>
<div align=center><img src="https://img.shields.io/github/v/release/MurthiNext/ScriptDirector"/>   <img src="https://img.shields.io/github/license/MurthiNext/ScriptDirector"/>   <img src="https://img.shields.io/github/stars/MurthiNext/ScriptDirector"/></div>

### &emsp;&emsp;这里是Script Director的ROCm-Version分支！即**AMD特供版**，通过使用ROCm强兼CUDA实现在AMD显卡上的硬件加速！由于MurthiNext没有高性能的AMD显卡用于测试，因此这个版本可能会出现各种问题，如果你执意要使用AMD显卡加速，请尝试自行修改代码并编译。
### &emsp;&emsp;此分支基于BETA-DEV，不与正式版同步更新。
### &emsp;&emsp;Script Director 是一个将音频文件与台本（文本）自动对齐，生成带时间戳字幕（SRT/LRC）的工具。它利用 **Faster Whisper** 进行语音识别，并通过 **Needleman-Wunsch** 风格的动态规划算法将识别结果与台本句子精确匹配，即使识别结果与台本不完全一致也能智能插值，确保每一句台本都有准确的时间码。

## 特性

- 🎙️ 基于 Faster Whisper 的高质量语音识别（通过 stable-whisper 封装，支持单词级时间戳）
- 📄 支持日语（`ja`）、中文（`zh`）、英文（`en`）等多语言台本分割（使用 `pysbd`）
- ✨ 短句模式：利用单词级时间戳，自动按标点分割长句，生成更精确的字幕
- 🔗 智能句子对齐：采用 Needleman-Wunsch 风格算法，处理插入、删除和替换
- ⚙️ 命令行界面（CLI）和配置文件支持，方便重复使用
- 🖥️ 图形化界面（GUI）支持（基于 customtkinter），提供更友好的操作体验
- 🎛️ 高级参数可配置：通过配置文件调整对齐惩罚、相似度偏移、默认时长、束搜索宽度、VAD（语音活动检测）等，适应不同场景
- 📊 实时进度反馈：GUI 中显示识别和对齐进度（0-100%），操作更直观
- 🤖 这篇README也是由DeepSeek V3.2写的，拜托，正经人谁会写文档嘛

## 安装

### 依赖
- Python 3.8+
- AMD ROCm 7.2 （若使用AMD显卡加速）
- 第三方Python库：`stable-ts`, `shutil`, `pysbd`, `rapidfuzz`, `click`（CLI 必须）, `customtkinter`（GUI 必须）

### 安装步骤
1. 克隆或下载本项目。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 下载 Faster Whisper 模型并解压到本地目录。

#### 以下列出主流Faster Whisper模型的下载地址。
* 快上加快
   * [Faster Whisper Large V3 Turbo](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
* 极致精度 
   * [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3)
   * [Faster Whisper Large V2](https://huggingface.co/Systran/faster-whisper-large-v2)
* 性能取舍
   * [Faster Whisper Medium](https://huggingface.co/Systran/faster-whisper-medium)
   * [Faster Whisper Small](https://huggingface.co/Systran/faster-whisper-small)
   * [Faster Whisper Tiny](https://huggingface.co/Systran/faster-whisper-tiny)

以上超链接均指向[Huggingface](https://huggingface.co/)，若你身处中国大陆无法访问，可尝试使用[HF-Mirror](https://hf-mirror.com/)此镜像网站。

## 使用方法

Script Director 提供两种使用方式：**命令行工具** `cli.py` 和 **图形界面** `gui.py`。

若你使用打包好的Release版本，以下内容的`python cli.py`均可替换为`.\cli.exe`。

### 1. 命令行工具（CLI）

CLI 支持子命令，便于配置和处理。

#### 首次运行配置
首次使用需通过 `init` 命令创建配置文件 `config.ini`：
```bash
python cli.py init
```
按照提示输入：
- **Faster Whisper 本地模型路径**：模型文件夹的路径（如 `./faster-whisper-large-v3`）
- **台本与音频所使用的语言代码**：例如 `ja`（日语）、`zh`（中文）、`en`（英文）
- **设备类型**：`cuda` 或 `cpu`
- **计算类型**：`float16`（GPU）、`int8`（CPU）等

**高级参数（可选）**：在初始化过程中，还可以设置以下高级参数（直接回车跳过则使用默认值）：
- `gap_penalty`：对齐惩罚值，默认 `-10`
- `similarity_offset`：相似度偏移，默认 `50`
- `default_duration`：默认字幕时长（秒），默认 `5.0`
- `max_combine`：最大合并片段数，默认 `5`
- `beam_size`：束搜索宽度，默认 `5`
- `vad_filter`：启用语音活动检测，默认 `False`
- `vad_parameters`：VAD 参数（JSON 格式），默认 `{}`

这些参数会自动写入配置文件的 `[advanced]` 节，便于后续调优。 有关配置信息的详细介绍请看下文。

#### 修改配置
如需修改配置项，可使用 `config` 命令：
```bash
python cli.py config model=/new/model/path
python cli.py config lang=en
```
支持修改的键（包括高级参数）：`model`, `lang`, `device`, `compute`, `gap_penalty`, `similarity_offset`, `default_duration`, `max_combine`, `beam_size`, `vad_filter`, `vad_parameters`。

#### 处理音频与台本
使用 `process` 命令生成字幕：
```bash
python cli.py process "音频文件路径,台本文件路径" [-t srt|lrc] [-n 自定义名称] [-p] [-s]
```
- 参数 `INPUT_STR` 必须用英文逗号分隔两个文件路径，程序会自动识别音频文件和台本文件（台本文件扩展名需为 `.txt`，音频文件支持常见格式）。
- `-t, --type`：输出格式，可选 `srt` 或 `lrc`，默认为 `srt`。
- `-n, --name`：自定义输出文件名（不含扩展名），默认与音频文件同名。
- `-p, --preprocess`：启用台本预处理，自动删除空行和方括号内容。
- `-s, --shorter`：启用短句模式（按标点分割长句，生成更精确的字幕）。该模式利用单词级时间戳，将长句拆分为短句，适合台词密集的音频。

**示例**：
```bash
# 基本用法
python cli.py process "meeting.wav,transcript.txt" -t lrc -n meeting_lyrics

# 启用预处理和短句模式
python cli.py process "audio.mp3,script.txt" -t srt -p -s
```
输出文件将保存在音频文件所在目录，名为 `meeting_lyrics.lrc` 或 `audio.srt`。

### 2. 图形界面（GUI）

如果你更喜欢可视化操作，可以直接运行图形界面：
```bash
python gui.py
```
界面采用左右分屏布局：
- **左侧**：配置与输入区域，包括模型路径、语言代码、设备类型、计算类型、音频/台本文件选择、输出名称、输出格式、预处理选项以及开始按钮。
- **右侧**：上方为进度条（实时显示识别和对齐进度，0-100%），下方为运行日志区域（实时显示所有日志输出）。

**操作步骤**：
1. 选择本地 Faster Whisper 模型文件夹
2. 选择音频文件（支持 .wav, .mp3, .flac, .m4a 等）
3. 选择台本文件（UTF-8 编码的 .txt 文件）
4. （可选）指定输出文件名（不含扩展名）
5. 选择输出格式（SRT 或 LRC）
6. 勾选“预处理台本”可自动清理空行和方括号标识
7. 勾选“短句模式”可启用按标点分割长句，生成更精确的字幕
8. 点击“开始处理”，实时查看进度和日志，处理完成后弹出提示

关闭窗口时会弹出确认框，若正在处理则询问是否强制退出，确保子进程被终止。

## 配置说明

`config.ini` 文件包含两个节：`[common]` 和 `[advanced]`。

### `[common]` 节（必填）
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

### `[advanced]` 节（可选）
```ini
[advanced]
gap_penalty = -10
similarity_offset = 50
default_duration = 5.0
max_combine = 5
beam_size = 5
vad_filter = False
vad_parameters = {}
```
- `gap_penalty`：序列对齐时的插入/删除惩罚值，负值，绝对值越大惩罚越重，影响匹配粒度。
- `similarity_offset`：相似度得分偏移量（原始相似度减去该值），用于将相似度得分映射到 DP 表的匹配得分，值越大匹配门槛越高。
- `default_duration`：当无法从 Whisper 结果中获取时间时，为未匹配句子分配的默认时长（秒）。
- `max_combine`：限制一句台本最多合并的 Whisper 片段数，防止过长合并影响精度。
- `beam_size`：束搜索宽度，影响识别速度和准确度，值越大越慢但越准。
- `vad_filter`：启用语音活动检测（Voice Activity Detection），可过滤非语音段，提高识别效率。
- `vad_parameters`：VAD 参数，JSON 格式，可自定义阈值等（默认为空，使用 Faster Whisper 内置参数）。

如果配置文件中缺少 `[advanced]` 节或某项参数，程序会使用默认值，不会报错。

## 项目结构
- `director.py`：核心模块，包含语音识别（基于 stable-whisper）、句子对齐、时间戳映射、字幕保存等功能。
- `cli.py`：命令行入口，处理参数、配置文件并调用 `director.direct_it`。
- `gui.py`：图形化界面入口，基于 customtkinter 实现，包含进度条和日志显示。
- `pre_process.py`：台本预处理模块，提供文本清洗功能（删除空行、方括号内容等）。

## 注意事项
- 音频格式支持取决于 Faster Whisper（常见格式如 `wav`, `mp3`, `m4a` 等）。
- 台本文件需为 UTF-8 编码的纯文本。
- 预处理功能会删除行首的方括号标识（如 `[人名]` 或 `【动作】`），但保留句子内部的方括号（如 `他说[笑]`）。
- 短句模式会启用单词级时间戳，识别速度略慢于普通模式，但字幕精确度更高，尤其适合台词密集或含有长句的音频。
- 如果 Whisper 识别结果与台本差异较大，可尝试调整 `[advanced]` 中的 `gap_penalty` 和 `similarity_offset` 参数，以获得更佳的对齐效果。
- 关闭 GUI 窗口时会强制终止所有子进程（包括 Faster Whisper 识别进程），确保程序完全退出。
- 多进程隔离机制可防止 Faster Whisper 底层崩溃导致主程序异常退出。
- 进度条分为识别（0-95%）和对齐（95-100%）两个阶段，反映实际耗时差异。

## 许可证
* 本项目采用 **MIT 许可证**。详情请参阅 [LICENSE](LICENSE) 文件。

---

**Happy Subtitling!** 🎬