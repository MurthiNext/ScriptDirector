import configparser
import json
import psutil
import re
import os
from typing import List, Tuple, Dict, Optional, Union, Sequence

from main_logger import logger

def load_config(config_path: str='config.ini') -> Dict:
    """
    读取配置，返回字典。
    如果配置文件存在，则读取并覆盖默认值；如果缺少某项，则使用默认值并记录警告。
    如果配置文件不存在，则使用全部默认值，并记录警告。
    """
    config = configparser.ConfigParser()
    common: dict[str, str] = {
        'model': '',
        'lang': 'ja',
        'device': 'cuda',
        'compute': 'float16',
    }
    advanced: dict[str, Union[int, float, bool, dict]] = {
        'gap_penalty': -10,
        'similarity_offset': 50,
        'default_duration': 5.0,
        'max_combine': 20,
        'beam_size': 5,
        'vad_filter': False,
        'vad_parameters': {},
    }
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
        if config.has_section('common'): # 读取 common 部分
            for key in common.keys():
                if config.has_option('common', key):
                    common[key] = config.get('common', key)
                else:
                    logger.warning(f"配置文件 {config_path} 中 [common] 节缺少 {key} 项，使用默认值 '{common[key]}'。")
        else:
            logger.warning(f"配置文件 {config_path} 中缺少 [common] 节，请确认无误。")
        if config.has_section('advanced'): # 读取 advanced 部分
            for key in advanced.keys():
                if config.has_option('advanced', key):
                    content =  config.get('advanced', key)
                    if isinstance(advanced[key], int) and not isinstance(advanced[key], bool): # int==bool的来了
                        try:
                            advanced[key] = int(content)
                        except ValueError:
                            logger.warning(f"配置文件 {config_path} 中 [advanced] 节 {key} 项值 '{content}' 无法转换为整数，使用默认值 {advanced[key]}。")
                    elif isinstance(advanced[key], float):
                        try:
                            advanced[key] = float(content)
                        except ValueError:
                            logger.warning(f"配置文件 {config_path} 中 [advanced] 节 {key} 项值 '{content}' 无法转换为浮点数，使用默认值 {advanced[key]}。")
                    elif isinstance(advanced[key], bool):
                        advanced[key] = content.lower() in ('true', '1', 'yes')
                    elif isinstance(advanced[key], dict):
                        try:
                            advanced[key] = json.loads(content)
                        except json.JSONDecodeError as e:
                            logger.warning(f"配置文件 {config_path} 中 [advanced] 节 {key} 项值 '{content}' 无法解析为 JSON，使用默认值 {{}}。错误: {e}")
                else:
                    logger.info(f"配置文件 {config_path} 中 [advanced] 节缺少 {key} 项，使用默认值 '{advanced[key]}'。")
        else:
            logger.info(f"配置文件 {config_path} 中缺少 [advanced] 节，使用默认配置。")
    else:
        logger.warning(f"配置文件 {config_path} 不存在，全部使用默认设置。")
    return {**common, **advanced}

def log_alignment_mapping(
        script_sents: List[str],
        target_sents: List[str],
        alignment: Sequence[Tuple[Optional[int], Optional[Union[Tuple[int, int], int]]]],
        name_a: str = "完整句子",
        name_b: str = "散落的单词"
    ) -> None:
    """
    记录对齐映射关系，格式：
      完整句子 [台本编号] ↔ 索引范围 [范围] : 台本句子内容
          散落的单词: [范围] 单词文本
    现在 alignment 中的 target 可以是整数索引或元组范围。
    """
    # 建立 script_idx -> 对应的 target 列表（可能是范围或单个索引）
    script_to_target = {}
    for s_idx, t_info in alignment:
        if s_idx is not None and t_info is not None:
            if isinstance(t_info, tuple):
                # 范围
                script_to_target.setdefault(s_idx, []).append(t_info)
            else:
                # 单个索引，转为范围
                script_to_target.setdefault(s_idx, []).append((t_info, t_info))

    output_text = ''
    for s_idx in sorted(script_to_target.keys()):
        ranges = sorted(script_to_target[s_idx])
        # 合并相邻或重叠的范围
        merged_ranges = []
        for r in ranges:
            if not merged_ranges:
                merged_ranges.append(list(r))
            else:
                last = merged_ranges[-1]
                if r[0] <= last[1] + 1:
                    last[1] = max(last[1], r[1])
                else:
                    merged_ranges.append(list(r))
        merged_ranges = [tuple(r) for r in merged_ranges]

        idx_str_parts = []
        for r_start, r_end in merged_ranges:
            if r_start == r_end:
                idx_str_parts.append(str(r_start))
            else:
                idx_str_parts.append(f"{r_start}-{r_end}")
        idx_str = ", ".join(idx_str_parts)

        sent_preview = script_sents[s_idx][:80] + "..." if len(script_sents[s_idx]) > 80 else script_sents[s_idx]
        output_text += f"  {name_a} [{s_idx}] ↔ 索引 [{idx_str}] : {sent_preview}\n"

        # 收集每个范围对应的单词文本
        words_detail = []
        for r_start, r_end in merged_ranges:
            texts = [target_sents[i][:50] + "..." if len(target_sents[i]) > 50 else target_sents[i] for i in range(r_start, r_end+1)]
            if r_start == r_end:
                words_detail.append(f"[{r_start}] {texts[0]}")
            else:
                words_detail.append(f"[{r_start}-{r_end}] {', '.join(texts)}")
        output_text += f"      {name_b}: {', '.join(words_detail)}\n"
    logger.info(f"\n========== 对齐映射（{name_a} ↔ {name_b}） ==========\n\n"+output_text+"\n" + "=" * 50)

def format_time_srt(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

def format_time_lrc(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds % 60
    hundredths = int((secs - int(secs)) * 100)
    return f"[{minutes:02d}:{int(secs):02d}.{hundredths:02d}]"

def save_srt(subtitles: List[Tuple[str, float, float]], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, (text, start, end) in enumerate(subtitles, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time_srt(start)} --> {format_time_srt(end)}\n")
            f.write(f"{text}\n\n")
    logger.info(f"已保存 SRT 字幕到 {output_path}")

def save_lrc(subtitles: List[Tuple[str, float, float]], output_path: str) -> None:
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, start, _ in subtitles:
            f.write(f"{format_time_lrc(start)} {text}\n")
    logger.info(f"已保存 LRC 歌词到 {output_path}")

def kill_process_tree(pid: Optional[int]) -> None:
    """
    递归终止进程及其所有子进程。
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        gone, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass

def is_bracket_line(line: str) -> bool:
    """
    判断整行是否仅由方括号内容（可能带空格）组成。
    """
    stripped = line.strip()
    return bool(re.match(r'^(\[[^\]]*\]|【[^】]*】)$', stripped))

def remove_line_brackets(line: str) -> str:
    """
    匹配行首的连续方括号标识（可包含前导空格）
    例如："  [角色] 文本" -> "文本"
    """
    pattern = r'^(\s*(\[[^\]]*\]|【[^】]*】)\s*)+'
    return re.sub(pattern, '', line)

def clean_script_text(text: str) -> str:
    """
    对台本全文进行清洗：
    1. 按行分割
    2. 删除空行
    3. 删除整行都是方括号的行（角色标识行）
    4. 删除行首的方括号标识（保留文本）
    5. 保留句子中的方括号
    """
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        # 删除空行
        if not line.strip():
            continue
        # 如果是整行都是方括号，跳过该行
        if is_bracket_line(line):
            continue
        # 删除行首的方括号标识
        line = remove_line_brackets(line)
        # 如果删除后为空行，跳过
        if not line.strip():
            continue
        cleaned_lines.append(line.strip())
    return '\n'.join(cleaned_lines)

def preprocess_file(input_path: str, output_path: Optional[str] = None) -> str:
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    cleaned = clean_script_text(content)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned)
    return cleaned

def interpolate_timestamps(
        time_map: Dict[int, Tuple[float, float]],
        total_sents: int,
        default_duration: float = 5.0
    ) -> List[Tuple[int, float, float]]:
    """
    根据已匹配句子的时间映射，为所有句子（包括未匹配的）插值生成时间。
    返回列表，每个元素为 (句子索引, 开始时间, 结束时间)
    """
    # 预处理：构建前驱和后继匹配索引数组
    prev_match: List[Optional[int]] = [None] * total_sents
    next_match: List[Optional[int]] = [None] * total_sents

    last_match = None
    for idx in range(total_sents):
        if idx in time_map:
            last_match = idx
        prev_match[idx] = last_match

    next_match_val = None
    for idx in range(total_sents - 1, -1, -1):
        if idx in time_map:
            next_match_val = idx
        next_match[idx] = next_match_val

    result = []
    for idx in range(total_sents):
        if idx in time_map:
            start, end = time_map[idx]
            result.append((idx, start, end))
        else:
            prev_idx = prev_match[idx]
            next_idx = next_match[idx]

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