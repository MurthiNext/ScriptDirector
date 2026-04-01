import os
import re
import logging
from rapidfuzz import fuzz
from typing import List, Tuple, Optional, Any
import multiprocessing

# 导入 director 中的相关函数和模块
from director import (
    logger, split_sentences_pysbd, log_alignment_mapping,
    save_srt, save_lrc,
    load_advanced_config
)

def align_sentence_lists(script_sents: List[str], whisper_sents: List[str], 
                                 gap_penalty: int = -10, similarity_offset: int = 50) -> List[Tuple[Optional[int], Optional[int]]]:
    """
    旧版对齐函数，返回单个单词索引，供只对齐模式使用。
    由于不需要合并，max_combine 参数已移除。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    logger.info(f"开始对齐（旧版）：台本 {n} 句，字幕 {m} 句")

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

def map_timestamps(alignment: List[Tuple[Optional[int], Optional[int]]], script_sents: List[str],
                   whisper_segments: List[Any], default_duration: float = 5.0, max_combine: int = 5,
                   progress_queue: Optional[multiprocessing.Queue] = None) -> List[Tuple[str, float, float]]: # 主干逻辑：对齐时间轴
    """
    原有的，按段落对齐的函数。（主函数已更换为按词对齐。）

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
            logger.warning(f"句子 {script_idx} 匹配了 {len(whisper_idxs)} 个片段，超过 max_combine={max_combine}，仅保留前 {max_combine} 个")
            whisper_idxs = whisper_idxs[:max_combine]
        start_time = whisper_segments[whisper_idxs[0]].start
        end_time = whisper_segments[whisper_idxs[-1]].end
        time_map[script_idx] = (start_time, end_time)
        logger.debug(f"句子 {script_idx} 时间: {start_time:.2f} -> {end_time:.2f}")

    # 第三步：为所有台本句子生成最终列表（包括未匹配的，通过插值）
    result = []
    total_sents = len(script_sents)
    for idx, script_idx in enumerate(range(total_sents)):
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
                end = default_duration
                logger.warning("无任何参考时间，使用默认值 0-5 秒")

            result.append((text, start, end))
            logger.debug(f"插值句子 {script_idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")

        # 发送对齐进度 (95% 到 100%)
        if progress_queue is not None:
            progress = 95 + (idx + 1) / total_sents * 5
            progress_queue.put(int(progress))

    logger.info(f"最终生成 {len(result)} 条字幕")
    return result

def parse_srt_file(filepath: str) -> List[Tuple[str, float, float]]:
    """解析 SRT 文件，返回 (文本, 开始时间, 结束时间) 列表"""
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
    return segments

def parse_lrc_file(filepath: str) -> List[Tuple[str, float, float]]:
    """解析 LRC 文件，返回 (文本, 开始时间, 结束时间) 列表（结束时间设为下一句的开始或默认值）"""
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
    return segments

def parse_subtitle_file(filepath: str) -> List[Tuple[str, float, float]]:
    """根据扩展名解析字幕文件"""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.srt':
        return parse_srt_file(filepath)
    elif ext == '.lrc':
        return parse_lrc_file(filepath)
    else:
        raise ValueError(f"不支持的字幕格式: {ext}")

def align_only(script_path: str, subtitle_path: str, output_path: str,
               output_format: str = 'srt', preprocess: bool = False,
               short_sentences: bool = False, config_path: str = 'config.ini') -> None:
    """
    只对齐模式：将台本与已有字幕文件对齐，生成新字幕。
    注意：此模式下 short_sentences 参数会被忽略，因为已有字幕不包含单词级时间戳，
    无法进行标点级分割。如果用户启用了短句模式，函数会记录警告并自动禁用。
    参数：
        script_path: 台本文件路径
        subtitle_path: 已有字幕文件路径（SRT或LRC）
        output_path: 输出文件路径
        output_format: 输出格式（'srt' 或 'lrc'）
        preprocess: 是否预处理台本
        short_sentences: 是否启用短句模式（在此模式下将被忽略）
        config_path: 配置文件路径
    """
    # 警告：短句模式在只对齐时无效
    if short_sentences:
        logger.warning("只对齐模式下不支持短句模式（单词级时间戳），已自动禁用短句模式。")
        short_sentences = False  # 强制禁用

    # 读取台本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    if preprocess:
        from pre_process import clean_script_text
        script_text = clean_script_text(script_text)
        logger.info("已对台本进行预处理（删除空行和方括号内容）")
    logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符")
    # 分割台本（不使用短句模式，因为短句模式需要单词级时间戳）
    if short_sentences:
        # 虽然用户要求，但我们强制禁用
        logger.warning("已忽略短句模式，使用普通句子分割。")
        script_sents = split_sentences_pysbd(script_text)
    else:
        script_sents = split_sentences_pysbd(script_text)
    logger.info(f"台本分割为 {len(script_sents)} 个句子")

    # 读取已有字幕
    subtitle_segments = parse_subtitle_file(subtitle_path)
    # 构造 whisper_segments 对象（具有 start/end 属性）
    class Segment:
        def __init__(self, start, end):
            self.start = start
            self.end = end
    whisper_segments = [Segment(start, end) for _, start, end in subtitle_segments]
    whisper_texts = [text for text, _, _ in subtitle_segments]

    # 读取高级配置
    advanced = load_advanced_config(config_path)
    gap_penalty = advanced['gap_penalty']
    similarity_offset = advanced['similarity_offset']
    default_duration = advanced['default_duration']

    # 对齐
    alignment = align_sentence_lists(script_sents, whisper_texts, gap_penalty, similarity_offset)
    log_alignment_mapping(script_sents, whisper_texts, alignment, "台本", "已有字幕")
    subtitles = map_timestamps(alignment, script_sents, whisper_segments, default_duration)

    # 保存结果
    if output_format == 'lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)
    logger.info("只对齐模式完成")