# Script Director BETA-DEV
<div align=left>
   <img src="https://img.shields.io/github/v/release/MurthiNext/ScriptDirector"/>
   <img src="https://img.shields.io/github/license/MurthiNext/ScriptDirector"/>
   <img src="https://img.shields.io/github/stars/MurthiNext/ScriptDirector"/>
</div>

### 这里是Script Director的抢先体验版！一般来讲，这个分支里存放着MurthiNext正在编写的内容，其中可能包含大量未经检验的代码，包括但不限于：从网上摘抄的内容、无法运行的代码、不存在的包或库，以及部分“脑袋一热”写出来的玩意，如果出现任何问题，请回到main分支！
### Script Director 是一个将音频文件与台本（文本）自动对齐，生成带时间戳字幕（SRT/LRC）的工具。它利用 **Faster Whisper** 进行语音识别，并通过 **Needleman-Wunsch** 风格的动态规划算法将识别结果与台本句子精确匹配，即使识别结果与台本不完全一致也能智能插值，确保每一句台本都有准确的时间码。
### 为了方便维护，此README不单独提供使用方法。