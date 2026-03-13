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
    对于未匹配的台本句子，根据前后已匹配句子的时间进行线性插值。
    返回列表，每个元素为 (句子文本, 开始时间, 结束时间)
    """
    # 1.建立 script_idx -> 对应的 whisper_idx 列表（仅当有匹配时）
    script_to_whisper = {}
    for script_idx, whisper_idx in alignment:
        if script_idx is not None and whisper_idx is not None:
            if script_idx not in script_to_whisper:
                script_to_whisper[script_idx] = []
            script_to_whisper[script_idx].append(whisper_idx)

    # 2.处理所有有匹配的句子，合并连续 whisper 索引
    matched_indices = sorted(script_to_whisper.keys())
    time_map = {} # script_idx -> (start, end)
    for script_idx in matched_indices:
        whisper_idxs = script_to_whisper[script_idx]
        # whisper_idxs 是按顺序排列的（因为 alignment 是按时间顺序的）
        start_time = whisper_segments[whisper_idxs[0]].start
        end_time = whisper_segments[whisper_idxs[-1]].end
        time_map[script_idx] = (start_time, end_time)

    # 3.为所有台本句子生成最终列表（包括未匹配的，通过插值）
    result = []
    for script_idx in range(len(script_sents)):
        text = script_sents[script_idx]
        if script_idx in time_map:
            # 直接匹配
            start, end = time_map[script_idx]
            result.append((text, start, end))
        else:
            # 未匹配，需要插值
            # 找到前后最近的已匹配句子
            prev_idx = None
            next_idx = None
            for i in range(script_idx - 1, -1, -1):
                if i in time_map:
                    prev_idx = i
                    break
            for i in range(script_idx + 1, len(script_sents)):
                if i in time_map:
                    next_idx = i
                    break

            if prev_idx is not None and next_idx is not None:
                # 前后都有匹配，线性插值
                prev_start, prev_end = time_map[prev_idx]
                next_start, next_end = time_map[next_idx]
                # 计算从 prev_end 到 next_start 的时间区间
                total_gap = next_start - prev_end
                # 计算需要插值的句子数量（包括当前）
                gap_sentences = next_idx - prev_idx - 1  # 中间缺失的句子数
                if gap_sentences > 0:
                    # 每个缺失句子分配的时间段长度
                    seg_duration = total_gap / (gap_sentences + 1)
                    # 当前句子是第几个缺失（从1开始）
                    offset = script_idx - prev_idx
                    start = prev_end + seg_duration * offset
                    end = start + seg_duration
                else:
                    # 理论上不会发生，但以防万一
                    start = prev_end
                    end = next_start
            elif prev_idx is not None:
                # 只有前一句
                prev_start, prev_end = time_map[prev_idx]
                # 使用前一句的时长作为参考
                duration = prev_end - prev_start
                start = prev_end
                end = start + duration
            elif next_idx is not None:
                # 只有后一句
                next_start, next_end = time_map[next_idx]
                duration = next_end - next_start
                end = next_start
                start = end - duration
            else:
                # 没有任何匹配句子（极端情况），使用默认时间
                start = 0.0
                end = 5.0

            result.append((text, start, end))

    # 按 script_idx 顺序 result 已经天然有序，无需再排序
    return result

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