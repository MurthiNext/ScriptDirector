import re
import os
from rapidfuzz import fuzz
from typing import List, Tuple, Optional, Any

from director import split_text_by_pysbd
from just_utils import (
    interpolate_timestamps,
    log_alignment_mapping,
    save_srt,
    save_lrc,
    load_config
)
from main_logger import logger

def parse_srt_file(filepath: str) -> List[Tuple[str, float, float]]:
    """
    解析 SRT 文件，返回 (文本, 开始时间, 结束时间) 列表。
    时间格式为 "HH:MM:SS,mmm"，转换为秒数。
    """
    logger.info(f"正在解析 SRT 文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    pattern = re.compile(
        r'\d+\n'
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n'
        r'(.*?)(?:\n\n|$)', re.DOTALL
    )
    segments = []
    for match in pattern.finditer(content):
        start_str, end_str, text = match.groups()
        start = sum(x * int(t) for x, t in zip([3600, 60, 1, 0.001], re.split('[:,]', start_str)))
        end = sum(x * int(t) for x, t in zip([3600, 60, 1, 0.001], re.split('[:,]', end_str)))
        text = text.replace('\n', ' ').strip()
        segments.append((text, start, end))
    logger.info(f"SRT 文件解析完成，提取到 {len(segments)} 条字幕。")
    return segments

def parse_lrc_file(filepath: str) -> List[Tuple[str, float, float]]:
    """
    解析 LRC 文件，返回 (文本, 开始时间, 结束时间) 列表。
    结束时间通过下一个时间戳推断，最后一个字幕默认持续 3 秒。
    时间格式为 "[MM:SS.xx]"，转换为秒数。
    """
    logger.info(f"正在解析 LRC 文件: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    segments = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'\[(\d{2}):(\d{2})\.(\d{2})\]\s*(.*)', line)
        if match:
            minutes, seconds, hundredths, text = match.groups()
            start = int(minutes) * 60 + int(seconds) + int(hundredths) / 100
            segments.append((text, start, 0.0))
    # 补全结束时间
    for i in range(len(segments) - 1):
        _, start, _ = segments[i]
        _, next_start, _ = segments[i+1]
        segments[i] = (segments[i][0], start, next_start)
    if segments:
        last_text, last_start, _ = segments[-1]
        segments[-1] = (last_text, last_start, last_start + 3.0)
    logger.info(f"LRC 文件解析完成，提取到 {len(segments)} 条字幕。")
    return segments

def parse_subtitle_file(filepath: str) -> List[Tuple[str, float, float]]:
    """
    根据扩展名解析字幕文件。
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.srt':
        return parse_srt_file(filepath)
    elif ext == '.lrc':
        return parse_lrc_file(filepath)
    else:
        raise ValueError(f"不支持的字幕格式: {ext}")

def align_sentence_lists_legacy(
        script_sents: List[str],
        whisper_sents: List[str], 
        gap_penalty: int = -10,
        similarity_offset: int = 50
    ) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    原有的，按段落对齐的函数。（主函数已更换为按词对齐。）
    由于对齐逻辑稍微简陋，返回的结果需要经build_timestamps_from_alignment函数构建时间戳映射。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp: List[List[float | int]] = [[0] * (m + 1) for _ in range(n + 1)]

    logger.info(f"开始对齐（旧版）：台本 {n} 句，字幕 {m} 句。")

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim_score = fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - similarity_offset
            match_score = dp[i-1][j-1] + sim_score
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match_score, delete, insert)

    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - similarity_offset):
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
    logger.info(f"对齐完成（旧版），路径长度 {len(alignment)}")
    return alignment

def build_timestamps_from_alignment(
        alignment: List[Tuple[Optional[int], Optional[int]]],
        script_sents: List[str],
        whisper_segments: List[Any], 
        default_duration: float = 5.0, 
        max_combine: int = 20
    ) -> List[Tuple[str, float, float]]:
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
        if len(whisper_idxs) > max_combine:
            logger.warning(f"句子 {script_idx} 匹配了 {len(whisper_idxs)} 个片段，超过 max_combine={max_combine}，仅保留前 {max_combine} 个。")
            whisper_idxs = whisper_idxs[:max_combine]
        start_time = whisper_segments[whisper_idxs[0]].start
        end_time = whisper_segments[whisper_idxs[-1]].end
        time_map[script_idx] = (start_time, end_time)

    # 第三步：使用公共插值函数
    interpolated = interpolate_timestamps(time_map, len(script_sents), default_duration)
    result = []
    for idx, start, end in interpolated:
        text = script_sents[idx]
        result.append((text, start, end))

    logger.info(f"最终生成 {len(result)} 条字幕")
    return result

def align_it(
        script_path: str,
        subtitle_path: str,
        output_path: str,
        output_format: str = 'srt',
        preprocess: bool = False,
        short_sentences: bool = False,
        config_path: str = 'config.ini'
    ) -> None:
    """
    只对齐模式：将台本与已有字幕文件对齐，生成新字幕。
    注意：此模式下 short_sentences 参数会被忽略，因为已有字幕不包含单词级时间戳，
    无法进行标点级分割。如果用户启用了短句模式，函数会记录警告并自动禁用。
    """
    if short_sentences:
        logger.warning("只对齐模式下不支持短句模式（单词级时间戳），已自动禁用短句模式。")
        short_sentences = False

    # 读取台本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    if preprocess:
        from just_utils import clean_script_text
        script_text = clean_script_text(script_text)
        logger.info("已对台本进行预处理（删除空行和方括号内容）")
    logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符。")
    # 分割台本
    if short_sentences:
        logger.warning("已忽略短句模式，使用普通句子分割。")
        script_sents = split_text_by_pysbd(script_text)
    else:
        script_sents = split_text_by_pysbd(script_text)
    logger.info(f"台本分割为 {len(script_sents)} 个句子。")

    # 读取已有字幕
    subtitle_segments = parse_subtitle_file(subtitle_path)
    # 构造 whisper_segments 对象
    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    whisper_segments = [Segment(start, end) for _, start, end in subtitle_segments]
    whisper_texts = [text for text, _, _ in subtitle_segments]

    # 读取高级配置
    settings = load_config(config_path)
    gap_penalty = settings['gap_penalty']
    similarity_offset = settings['similarity_offset']
    default_duration = settings['default_duration']

    # 对齐
    alignment = align_sentence_lists_legacy(script_sents, whisper_texts, gap_penalty, similarity_offset)
    log_alignment_mapping(script_sents, whisper_texts, alignment, "台本", "已有字幕")
    subtitles = build_timestamps_from_alignment(alignment, script_sents, whisper_segments, default_duration)

    # 保存结果
    if output_format == 'lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)
    logger.info("只对齐模式完成。")