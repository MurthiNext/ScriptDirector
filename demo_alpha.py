import os
import pysbd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

def format_time_srt(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

def format_time_lrc(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    hundredths = int((secs - int(secs)) * 100)
    return f"[{minutes:02d}:{int(secs):02d}.{hundredths:02d}]"

def save_srt(subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, (text, start, end) in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
            f.write(f"{text}\n\n")

def save_lrc(subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, start, _ in subtitles:
            f.write(f"{format_time_lrc(start)} {text}\n")

def split_sentences_pysbd(text, language='ja'):
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    return [s.strip() for s in sentences if s.strip()]

def align_sentence_lists(script_sents, whisper_sents, gap_penalty=-10):
    """
    使用 Needleman-Wunsch 风格的对齐算法，对齐两个句子列表。
    返回对齐路径列表，每个元素为 (script_idx, whisper_idx)，允许 None 表示插入/删除。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    # 初始化边界
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty

    # 填充 DP 表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 相似度得分映射到 [-50, 50]
            sim_score = fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - 50
            match = dp[i-1][j-1] + sim_score
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    # 回溯
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - 50):
            alignment.append((i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            alignment.append((i-1, None))
            i -= 1
        else:
            alignment.append((None, j-1))
            j -= 1
    alignment.reverse()
    return alignment

def map_timestamps(alignment, script_sents, whisper_segments):
    """
    根据对齐路径，为每个匹配的台本句子分配时间戳。
    返回列表，每个元素为 (句子文本, 开始时间, 结束时间)
    """
    mapped = []
    i = 0
    while i < len(alignment):
        script_idx, whisper_idx = alignment[i]
        if script_idx is not None and whisper_idx is not None:
            # 匹配上的台本句子
            sent = script_sents[script_idx]
            start_time = whisper_segments[whisper_idx].start
            end_time = whisper_segments[whisper_idx].end
            # 向后看是否还有连续的 whisper 句子属于同一个 script 句子
            while i + 1 < len(alignment) and alignment[i+1][0] == script_idx + 1 and alignment[i+1][1] is not None:
                i += 1
                end_time = whisper_segments[alignment[i][1]].end
            mapped.append((sent, start_time, end_time))
        # 忽略没有匹配上的句子（插入或删除）
        i += 1
    return mapped

def main(audio_path, script_path, output_path, local_model_path, language='ja', device='cuda', compute_type='float16'):
    # 1. 加载 Whisper 模型
    model = WhisperModel(local_model_path, device=device, compute_type=compute_type)

    # 2. 转录音频，获取带时间戳的片段
    print("正在转录音频...")
    segments, info = model.transcribe(audio_path, language=language, word_timestamps=False)
    whisper_segments = list(segments)
    print(f"Whisper 识别出 {len(whisper_segments)} 个片段")

    # 3. 读取台本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()

    # 4. 使用 pysbd 分割台本句子
    script_sents = split_sentences_pysbd(script_text, language=language)
    print(f"台本共分割为 {len(script_sents)} 个句子")

    # 5. 准备 Whisper 句子列表（直接使用每个片段的文本）
    whisper_sents = [seg.text for seg in whisper_segments]

    # 6. 对齐句子列表
    print("正在对齐句子...")
    alignment = align_sentence_lists(script_sents, whisper_sents)

    # 7. 映射时间戳
    subtitles = map_timestamps(alignment, script_sents, whisper_segments)
    print(f"成功对齐 {len(subtitles)} 条字幕")

    # 8. 根据输出文件后缀选择保存格式
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.lrc':
        save_lrc(subtitles, output_path)
        print(f"LRC 歌词已保存至 {output_path}")
    else:
        # 默认保存为 SRT
        save_srt(subtitles, output_path)
        print(f"SRT 字幕已保存至 {output_path}")

if __name__ == "__main__":
    main(
        audio_path="audio.wav",# 音频文件路径
        script_path="script.txt",# 台本文件路径
        output_path="output.lrc",# 输出文件路径（.srt 或 .lrc）
        local_model_path="./faster-whisper-large-v3-turbo",# Faster Whisper 本地模型路径
        language='ja',# 语言
        device='cuda',# 计算设备
        compute_type='float16'# 计算类型
    )