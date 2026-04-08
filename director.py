import os
import logging
import pysbd
import stable_whisper
from rapidfuzz import fuzz
import multiprocessing
import traceback
import time
from typing import List, Tuple, Optional, Union, Any
from logging.handlers import QueueHandler
import configparser
import json
import re
import psutil

from timeline import interpolate_timestamps

__author__ = 'MurthiNext'
__version__ = '2.1.5 Beta'
__date__ = '2026/04/08'

# 进度相关常量
PROGRESS_TRANSCRIBE_MAX = 95
PROGRESS_ALIGN_START = 95
PROGRESS_ALIGN_END = 99
PROGRESS_DONE = 100
# 进程超时设置
PROCESS_TIMEOUT = 3600

# 编译正则表达式
_PUNCT_SPLIT_RE = re.compile(r'(?<=[。！？…、．])\s*')

if os.path.isfile('log.log'):
    with open('log.log', 'w', encoding='utf-8') as wf:
        wf.write('')
logger = logging.getLogger('director')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler('log.log', encoding='utf-8') # 文件处理器
    ch = logging.StreamHandler() # 终端处理器
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def load_advanced_config(config_path='config.ini'):
    """读取 [advanced] 节的配置，返回字典，未设置的项使用默认值"""
    config = configparser.ConfigParser()
    defaults = {
        'gap_penalty': '-10',
        'similarity_offset': '50',
        'default_duration': '5.0',
        'max_combine': '15',
        'beam_size': '5',
        'vad_filter': 'False',
        'vad_parameters': '{}',
    }
    if os.path.exists(config_path):
        config.read(config_path, encoding='utf-8')
        if config.has_section('advanced'):
            for key, default in defaults.items():
                if config.has_option('advanced', key):
                    defaults[key] = config.get('advanced', key)
    # 转换类型
    try:
        vad_params = json.loads(defaults['vad_parameters'])
    except json.JSONDecodeError as e:
        logger.warning(f"解析 vad_parameters 失败，使用默认值 {{}}。错误: {e}")
        vad_params = {}
    advanced = {
        'gap_penalty': int(defaults['gap_penalty']),
        'similarity_offset': int(defaults['similarity_offset']),
        'default_duration': float(defaults['default_duration']),
        'max_combine': int(defaults['max_combine']),
        'beam_size': int(defaults['beam_size']),
        'vad_filter': defaults['vad_filter'].lower() in ('true', '1', 'yes'),
        'vad_parameters': vad_params,
    }
    return advanced

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
        f.close()
    logger.info(f"已保存 SRT 字幕到 {output_path}")

def save_lrc(subtitles: List[Tuple[str, float, float]], output_path: str) -> None:
    subtitles = normalize_subtitles(subtitles)
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, start, _ in subtitles:
            f.write(f"{format_time_lrc(start)} {text}\n")
        f.close()
    logger.info(f"已保存 LRC 歌词到 {output_path}")

def normalize_subtitle_text(text: str) -> str:
    """删除字幕文本中的空行和内部换行，将多行合并为一行。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)

def normalize_subtitles(subtitles: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    normalized = []
    for text, start, end in subtitles:
        text = normalize_subtitle_text(text)
        if text:
            normalized.append((text, start, end))
    return normalized

def split_sentences_pysbd(text: str, language: str = 'ja') -> List[str]:
    text = text.replace('\r', ' ').replace('\n', ' ')
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    result = [s.strip() for s in sentences if s.strip()]
    logger.info(f"台本分割为 {len(result)} 个句子")
    for i, sent in enumerate(result, 1):
        logger.debug(f"句子 {i}: {sent[:50]}..." if len(sent) > 50 else f"句子 {i}: {sent}")
    return result

def align_sentence_lists(script_sents: List[str], whisper_sents: List[str], gap_penalty: int = -10, similarity_offset: int = 50, max_combine: int = 5) -> List[Tuple[Optional[int], Optional[Tuple[int, int]]]]:
    """
    使用 Needleman-Wunsch 风格的对齐算法，对齐两个句子列表。
    该版本为重构后的增强版本，允许一个台本句子匹配连续的多个 Whisper 句子（范围），以更好地处理台本与识别结果之间的差异。
    原版本存入only_align.py，供只对齐模式使用。

    返回对齐路径列表，每个元素为 (script_idx, whisper_range)，允许 None 表示插入/删除。
    whisper_range 是一个元组 (start_idx, end_idx)，表示连续的一段单词索引。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # 记录匹配的单词范围，用于回溯
    match_range = [[None] * (m + 1) for _ in range(n + 1)]

    logger.info(f"开始对齐：台本 {n} 句，当前字幕 {m} 句")

    # 初始化边界
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        # 匹配范围记录为删除
        match_range[i][0] = ('delete', i-1)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        match_range[0][j] = ('insert', j-1)

    # 填充 DP 表
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 1. 考虑匹配一个单词
            sim_single = fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - similarity_offset
            match_single = dp[i-1][j-1] + sim_single
            best_score = match_single
            best_range = (j-1, j-1)  # 单个单词范围
            best_type = 'match_single'

            # 2. 考虑匹配连续多个单词（最多 max_combine 个）
            max_len = min(max_combine, j)
            for length in range(2, max_len + 1):
                # 取连续 length 个单词：j-length 到 j-1
                start = j - length
                # 拼接单词文本
                combined_text = ' '.join(whisper_sents[start:j])
                sim = fuzz.token_set_ratio(script_sents[i-1], combined_text) - similarity_offset
                score = dp[i-1][start] + sim
                if score > best_score:
                    best_score = score
                    best_range = (start, j-1)
                    best_type = 'match_multi'

            # 3. 删除（台本句子未匹配任何单词）
            delete = dp[i-1][j] + gap_penalty
            if delete > best_score:
                best_score = delete
                best_type = 'delete'
                best_range = None

            # 4. 插入（Whisper 单词未匹配任何台本句子）
            insert = dp[i][j-1] + gap_penalty
            if insert > best_score:
                best_score = insert
                best_type = 'insert'
                best_range = None

            dp[i][j] = best_score
            if best_type == 'match_single':
                match_range[i][j] = ('match', (j-1, j-1))
            elif best_type == 'match_multi':
                match_range[i][j] = ('match', best_range)
            elif best_type == 'delete':
                match_range[i][j] = ('delete', i-1)
            else:  # insert
                match_range[i][j] = ('insert', j-1)

    # 回溯
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and match_range[i][j] is not None:
            typ, data = match_range[i][j]
            if typ == 'match':
                start_idx, end_idx = data
                alignment.append((i-1, (start_idx, end_idx)))
                logger.debug(f"匹配: 台本 {i-1} <-> Whisper 范围 [{start_idx}-{end_idx}]")
                i -= 1
                j = start_idx
                continue
            elif typ == 'delete':
                alignment.append((i-1, None))
                logger.debug(f"台本插入: {i-1}")
                i -= 1
                continue
            elif typ == 'insert':
                alignment.append((None, data))
                logger.debug(f"Whisper 删除: {data}")
                j -= 1
                continue
        if i > 0:
            alignment.append((i-1, None))
            i -= 1
        else:
            alignment.append((None, j-1))
            j -= 1
    alignment.reverse()
    logger.info(f"对齐完成，路径长度 {len(alignment)}")
    return alignment

def split_text_by_punctuation(text: str) -> List[str]:
    text = text.replace('\r', ' ').replace('\n', ' ')
    parts = _PUNCT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def is_punctuation_only(text: str) -> bool:
    """判断文本是否只包含标点符号和空白"""
    punct = set('。！？…．、，．？！；：""''（）【】《》')
    for ch in text.strip():
        if ch not in punct and not ch.isspace():
            return False
    return True

def log_alignment_mapping(script_sents: List[str], target_sents: List[str], alignment: List[Tuple[Optional[int], Optional[Union[int, Tuple[int, int]]]]], name_a: str = "完整句子", name_b: str = "散落的单词") -> None:
    """
    记录对齐映射关系，格式：
      完整句子 [台本编号] ↔ 索引范围 [范围] : 台本句子内容
          散落的单词: [范围] 单词文本, ...
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

    logger.info(f"========== 对齐映射（{name_a} ↔ {name_b}） ==========")
    output_text = '\n\n'
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
    logger.info(output_text)
    logger.info("=" * 50)

def _transcribe_unified(model, audio_path: str, language: str,
                        beam_size: int, vad_filter: bool, vad_parameters: dict,
                        progress_queue: Optional[multiprocessing.Queue]) -> Tuple[List[Tuple[str, float, float]], float]:
    """统一转录：返回单词列表（word, start, end）和总时长，同时发送进度，并实时记录识别片段"""
    logger.info("开始转录音频...")
    # 定义内部进度回调
    def progress_cb(p, eta):
        if progress_queue is not None and eta > 0:
            progress = int((p / eta) * PROGRESS_TRANSCRIBE_MAX)
            progress_queue.put(progress)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters if vad_filter else None,
        progress_callback=progress_cb
    )
    total_duration = result.ori_dict.get('duration')
    # 收集所有单词
    all_words = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                all_words.append((w.word.strip(), w.start, w.end))
    if total_duration is None and all_words:
        total_duration = all_words[-1][2]
    logger.info(f"识别完成，共 {len(all_words)} 个单词")
    if progress_queue is not None:
        progress_queue.put(PROGRESS_TRANSCRIBE_MAX)
    return all_words, total_duration

def _prepare_script(script_path: str, preprocess: bool, short_sentences: bool) -> Tuple[str, List[str]]:
    """读取并分割台本，返回原始文本和句子列表（如果短句模式，则进一步按标点拆分成短句）"""
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    if preprocess:
        from pre_process import clean_script_text
        script_text = clean_script_text(script_text)
        logger.info("已对台本进行预处理（删除空行和方括号内容）")
    logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符")
    if short_sentences:
        script_sents = split_text_by_punctuation(script_text)
        logger.info(f"按标点分割台本为 {len(script_sents)} 个短句")
    else:
        script_sents = split_sentences_pysbd(script_text)
    return script_text, script_sents

def _build_subtitles_from_words(script_sents: List[str], all_words: List[Tuple[str, float, float]],
                                gap_penalty: int, similarity_offset: int, default_duration: float,
                                max_combine: int, progress_queue: Optional[multiprocessing.Queue]) -> List[Tuple[str, float, float]]:
    """
    将台本句子与单词列表对齐，为每个句子分配时间戳。
    现在使用增强的对齐算法，允许一个台本句子匹配多个单词。
    """
    # 提取单词文本列表
    word_texts = [w[0] for w in all_words]
    # 对齐台本句子和单词序列（使用新的对齐函数，允许匹配范围）
    alignment = align_sentence_lists(script_sents, word_texts, gap_penalty, similarity_offset, max_combine)
    
    # 输出对齐日志（现在 alignment 中包含范围）
    log_alignment_mapping(script_sents, word_texts, alignment, "台本", "单词")
    
    # 构建时间映射：每个台本句子取匹配的单词范围的最小开始时间和最大结束时间
    time_map = {}
    for s_idx, t_info in alignment:
        if s_idx is not None and t_info is not None:
            if isinstance(t_info, tuple):
                start_idx, end_idx = t_info
            else:
                start_idx = end_idx = t_info
            start = all_words[start_idx][1]
            end = all_words[end_idx][2]
            time_map[s_idx] = (start, end)
            logger.debug(f"句子 {s_idx} 匹配单词范围 {start_idx}-{end_idx}, 时间 [{start:.2f}-{end:.2f}]")

    # 生成字幕（插值）
    interpolated = interpolate_timestamps(time_map, len(script_sents), default_duration)
    result = []
    for idx, start, end in interpolated:
        text = normalize_subtitle_text(script_sents[idx])
        if not text:
            continue
        result.append((text, start, end))
        logger.debug(f"句子 {idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")

    # 过滤纯标点行
    filtered = []
    for text, start, end in result:
        if not is_punctuation_only(text):
            filtered.append((text, start, end))
    filtered = normalize_subtitles(filtered)
    logger.info(f"生成 {len(filtered)} 条字幕（过滤后）")
    return filtered

def kill_process_tree(pid):
    """递归终止进程及其所有子进程"""
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

def _run_whisper_task(audio_path: str, script_path: str, output_path: str,
                      local_model_path: str, language: str, device: str,
                      compute_type: str, result_queue: multiprocessing.Queue,
                      preprocess: bool = False,
                      advanced: Optional[dict] = None,
                      log_queue: Optional[multiprocessing.Queue] = None,
                      progress_queue: Optional[multiprocessing.Queue] = None,
                      short_sentences: bool = False) -> None:
    """
    子进程执行的任务：加载模型、识别、对齐、生成字幕列表，并将结果放入队列。
    如果提供了 log_queue，则将日志也发送到该队列。
    如果提供了 progress_queue，则发送进度（0-100 整数）
    """
    try:
        if log_queue is not None:
            # 清除所有现有处理器，只保留 QueueHandler
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            queue_handler = QueueHandler(log_queue)
            logger.addHandler(queue_handler)

        if advanced is None:
            advanced = {}
        beam_size = advanced.get('beam_size', 5)
        vad_filter = advanced.get('vad_filter', False)
        vad_parameters = advanced.get('vad_parameters', {})
        gap_penalty = advanced.get('gap_penalty', -10)
        similarity_offset = advanced.get('similarity_offset', 50)
        default_duration = advanced.get('default_duration', 5.0)
        max_combine = advanced.get('max_combine', 5)

        logger.info(f"加载模型: {local_model_path}")
        model = stable_whisper.load_faster_whisper(local_model_path, device=device, compute_type=compute_type)

        # 统一转录，获取单词列表
        all_words, total_duration = _transcribe_unified(
            model, audio_path, language, beam_size, vad_filter, vad_parameters, progress_queue
        )

        # 准备台本句子
        _, script_sents = _prepare_script(script_path, preprocess, short_sentences)

        # 生成字幕（使用增强的对齐，允许匹配范围）
        subtitles = _build_subtitles_from_words(
            script_sents, all_words, gap_penalty, similarity_offset, default_duration, max_combine, progress_queue
        )

        result_queue.put(('result', subtitles))
        result_queue.close()
        result_queue.join_thread()
        time.sleep(0.5)
        logger.info("处理完成，结果已放回队列。")
        if progress_queue is not None:
            progress_queue.put(PROGRESS_ALIGN_END)

    except Exception as e:
        error_msg = f"子进程发生错误：{str(e)}\n{traceback.format_exc()}"
        result_queue.put(('error', error_msg))
        logger.error(error_msg)
    finally:
        pass

def direct_it(audio_path: str, script_path: str, output_path: str,
              local_model_path: str, language: str = 'ja',
              device: str = 'cuda', compute_type: str = 'float16',
              preprocess: bool = False,
              config_path: str = 'config.ini',
              log_queue: Optional[multiprocessing.Queue] = None,
              progress_queue: Optional[multiprocessing.Queue] = None,
              short_sentences: bool = False) -> None:
    advanced = load_advanced_config(config_path)

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_whisper_task,
        args=(audio_path, script_path, output_path, local_model_path,
              language, device, compute_type, result_queue,
              preprocess, advanced, log_queue, progress_queue, short_sentences)
    )
    p.start()
    logger.info("已启动子进程进行语音识别...")

    try:
        result = result_queue.get(timeout=PROCESS_TIMEOUT)
        if isinstance(result, tuple) and result[0] == 'error':
            logger.error(f"子进程返回错误信息: {result[1]}")
            raise RuntimeError(f"语音识别失败: {result[1]}")
        elif isinstance(result, tuple) and result[0] == 'result':
            subtitles = result[1]
        else:
            # 兼容旧格式（直接是列表）
            subtitles = result
    except Exception as e:
        if p.is_alive():
            logger.error("子进程可能卡死，正在终止...")
            kill_process_tree(p.pid)
            p.join()
        raise RuntimeError(f"获取结果失败: {e}")

    subtitles = [r for r in subtitles if r and r[0]]

    p.join(timeout=10)
    if p.is_alive():
        logger.warning("子进程未及时退出，强制终止")
        kill_process_tree(p.pid)
        p.join()

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)

    if progress_queue is not None:
        progress_queue.put(PROGRESS_DONE)
    logger.info("字幕文件保存完成。")

if __name__ == "__main__":
    direct_it(
        audio_path="audio.wav",                # 音频文件路径
        script_path="script.txt",               # 台本文件路径
        output_path="output.lrc",                # 输出文件路径（.srt 或 .lrc）
        local_model_path="./faster-whisper-large-v3-turbo",  # 本地模型文件夹路径
        language='ja',                           # 语言代码
        device='cuda',                           # 计算设备 'cuda' 或 'cpu'
        compute_type='float16',                   # 计算类型
        short_sentences=True                    # 启用短句模式
    )