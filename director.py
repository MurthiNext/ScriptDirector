import os
import logging
import pysbd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

# 配置日志：只写入文件，不输出到控制台
def setup_logger():
    logger = logging.getLogger('director')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # 文件处理器
        fh = logging.FileHandler('log.log', encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(e)
    return wrapper

logger = setup_logger()

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
    logger.info(f"已保存 SRT 字幕到 {output_path}")

def save_lrc(subtitles, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, start, _ in subtitles:
            f.write(f"{format_time_lrc(start)} {text}\n")
    logger.info(f"已保存 LRC 歌词到 {output_path}")

def split_sentences_pysbd(text, language='ja'):
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    result = [s.strip() for s in sentences if s.strip()]
    logger.info(f"台本分割为 {len(result)} 个句子")
    for i, sent in enumerate(result, 1):
        logger.debug(f"句子 {i}: {sent[:50]}..." if len(sent) > 50 else f"句子 {i}: {sent}")
    return result

@exception_handler
def align_sentence_lists(script_sents, whisper_sents, gap_penalty=-10):
    """
    使用 Needleman-Wunsch 风格的对齐算法，对齐两个句子列表。
    返回对齐路径列表，每个元素为 (script_idx, whisper_idx)，允许 None 表示插入/删除。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    logger.info(f"开始对齐：台本 {n} 句，Whisper {m} 句")

    # 初始化边界
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty

    # 填充 DP 表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
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
            logger.debug(f"匹配: 台本 {i-1} <-> Whisper {j-1}")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap_penalty:
            alignment.append((i-1, None))
            logger.debug(f"台本插入: {i-1}")
            i -= 1
        else:
            alignment.append((None, j-1))
            logger.debug(f"Whisper 删除: {j-1}")
            j -= 1
    alignment.reverse()
    logger.info(f"对齐完成，路径长度 {len(alignment)}")
    return alignment

@exception_handler
def map_timestamps(alignment, script_sents, whisper_segments):
    """
    根据对齐路径，为每个匹配的台本句子分配时间戳。
    对于未匹配的台本句子，根据前后已匹配句子的时间进行线性插值。
    返回列表，每个元素为 (句子文本, 开始时间, 结束时间)
    """
    # 第一步：建立 script_idx -> 对应的 whisper_idx 列表（仅当有匹配时）
    script_to_whisper = {}
    for script_idx, whisper_idx in alignment:
        if script_idx is not None and whisper_idx is not None:
            if script_idx not in script_to_whisper:
                script_to_whisper[script_idx] = []
            script_to_whisper[script_idx].append(whisper_idx)

    # 第二步：处理所有有匹配的句子，合并连续 whisper 索引
    matched_indices = sorted(script_to_whisper.keys())
    logger.info(f"直接匹配的句子数: {len(matched_indices)}")
    time_map = {}
    for script_idx in matched_indices:
        whisper_idxs = script_to_whisper[script_idx]
        start_time = whisper_segments[whisper_idxs[0]].start
        end_time = whisper_segments[whisper_idxs[-1]].end
        time_map[script_idx] = (start_time, end_time)
        logger.debug(f"句子 {script_idx} 时间: {start_time:.2f} -> {end_time:.2f}")

    # 第三步：为所有台本句子生成最终列表（包括未匹配的，通过插值）
    result = []
    for script_idx in range(len(script_sents)):
        text = script_sents[script_idx]
        if script_idx in time_map:
            start, end = time_map[script_idx]
            result.append((text, start, end))
            logger.debug(f"已匹配句子 {script_idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")
        else:
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
                prev_start, prev_end = time_map[prev_idx]
                next_start, next_end = time_map[next_idx]
                total_gap = next_start - prev_end
                gap_sentences = next_idx - prev_idx - 1
                if gap_sentences > 0:
                    seg_duration = total_gap / (gap_sentences + 1)
                    offset = script_idx - prev_idx
                    start = prev_end + seg_duration * offset
                    end = start + seg_duration
                else:
                    start = prev_end
                    end = next_start
            elif prev_idx is not None:
                prev_start, prev_end = time_map[prev_idx]
                duration = prev_end - prev_start
                start = prev_end
                end = start + duration
            elif next_idx is not None:
                next_start, next_end = time_map[next_idx]
                duration = next_end - next_start
                end = next_start
                start = end - duration
            else:
                start = 0.0
                end = 5.0
                logger.warning("无任何参考时间，使用默认值 0-5 秒")

            result.append((text, start, end))
            logger.debug(f"插值句子 {script_idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")

    logger.info(f"最终生成 {len(result)} 条字幕")
    return result

@exception_handler
def main(audio_path, script_path, output_path, local_model_path, language='ja', device='cuda', compute_type='float16'):
    # 1. 加载 Whisper 模型
    logger.info(f"加载模型: {local_model_path}")
    model = WhisperModel(local_model_path, device=device, compute_type=compute_type)

    # 2. 转录音频，获取带时间戳的片段
    logger.info("开始转录音频...")
    segments, info = model.transcribe(audio_path, language=language, word_timestamps=False)
    whisper_segments = list(segments)
    logger.info(f"Whisper 识别出 {len(whisper_segments)} 个片段")
    for idx, seg in enumerate(whisper_segments):
        logger.debug(f"片段 {idx}: [{seg.start:.2f}-{seg.end:.2f}] {seg.text}")

    # 3. 读取台本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符")

    # 4. 使用 pysbd 分割台本句子
    script_sents = split_sentences_pysbd(script_text, language=language)

    # 5. 准备 Whisper 句子列表（直接使用每个片段的文本）
    whisper_sents = [seg.text for seg in whisper_segments]

    # 6. 对齐句子列表
    alignment = align_sentence_lists(script_sents, whisper_sents)

    # 7. 映射时间戳
    subtitles = map_timestamps(alignment, script_sents, whisper_segments)

    # 8. 根据输出文件后缀选择保存格式
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)

    logger.info("处理完成")

if __name__ == "__main__":
    main(
        audio_path='./audio.wav',
        script_path='./script.txt',
        output_path='./output.lrc',
        local_model_path='./faster-whisper-large-v3-turbo',
        language='ja',
        device='cuda',
        compute_type='float16'
    )