import re
import os
from typing import List, Tuple, Dict

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

def interpolate_timestamps(
        time_map: Dict[int, Tuple[float, float]],
        total_sents: int,
        default_duration: float = 5.0
    ) -> List[Tuple[int, float, float]]:
    """
    根据已匹配句子的时间映射，为所有句子（包括未匹配的）插值生成时间。
    返回列表，每个元素为 (句子索引, 开始时间, 结束时间)
    """
    logger.info(f">正在运行时间轴差值算法(interpolate_timestamps)")
    logger.info(f">台本句子数为n={total_sents}，已匹配时间的句子数为{len(time_map)}。")
    logger.info(f">时间复杂度O(n²)，空间复杂度O(n)。")
    result = []
    for idx in range(total_sents):
        if idx in time_map:
            start, end = time_map[idx]
            result.append((idx, start, end))
        else:
            # 找到前后最近的已匹配句子
            prev_idx = None
            next_idx = None
            for i in range(idx - 1, -1, -1):
                if i in time_map:
                    prev_idx = i
                    break
            for i in range(idx + 1, total_sents):
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
                    offset = idx - prev_idx
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
            result.append((idx, start, end))
    return result