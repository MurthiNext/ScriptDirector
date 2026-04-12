import stable_whisper
import multiprocessing
import traceback
import time
import pysbd
import re
import os
import numpy as np
from typing import List, Tuple, Optional, Dict
from rapidfuzz import fuzz
from faster_whisper import WhisperModel
from stable_whisper.result import WhisperResult

from just_utils import (
    log_alignment_mapping,
    interpolate_timestamps,
    load_config,
    kill_process_tree,
    save_lrc,
    save_srt
)
from main_logger import logger, setup_logging, setup_subprocess_logging

__author__ = 'MurthiNext'
__version__ = '2.3.0 Beta'
__date__ = '2026/04/12'

# 进度相关常量
PROGRESS_TRANSCRIBE_MAX = 80
PROGRESS_ALIGN_START = 80
PROGRESS_ALIGN_END = 99
PRE_WEIGHT = 0.8
DP_WEIGHT = 0.2
PROGRESS_DONE = 100
# 进程超时设置
PROCESS_TIMEOUT = 3600
# 编译正则表达式
_PUNCT_SPLIT_RE = re.compile(r'(?<=[。！？…、．])\s*')

def normalize_subtitle_text(text: str) -> str:
    """
    删除字幕文本中的空行和内部换行，将多行合并为一行。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return ' '.join(lines)

def normalize_subtitles(subtitles: List[Tuple[str, float, float]]) -> List[Tuple[str, float, float]]:
    """
    批量处理字幕列表，剔除空行。
    """
    normalized = []
    for text, start, end in subtitles:
        text = normalize_subtitle_text(text)
        if text:
            normalized.append((text, start, end))
    return normalized

def split_text_by_pysbd(text: str, language: str = 'ja') -> List[str]:
    """
    根据pysbd分割文本。
    """
    text = text.replace('\r', ' ').replace('\n', ' ')
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    result = [s.strip() for s in sentences if s.strip()]
    logger.info(f"台本分割为 {len(result)} 个句子。")
    return result

def split_text_by_punctuation(text: str) -> List[str]:
    """
    根据标点符号分割文本。
    """
    text = text.replace('\r', ' ').replace('\n', ' ')
    parts = _PUNCT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]

def is_punctuation_only(text: str) -> bool:
    """
    判断文本是否只包含标点符号和空白。
    """
    punct = set('。！？…．、，．？！；：""''（）【】《》')
    for ch in text.strip():
        if ch not in punct and not ch.isspace():
            return False
    return True

def _align_sentence_lists(
        script_sents: List[str],
        whisper_sents: List[str],
        gap_penalty: int = -10,
        similarity_offset: int = 50,
        max_combine: int = 20,
        progress_queue: Optional[multiprocessing.Queue] = None
    ) -> List[Tuple[Optional[int], Optional[Tuple[int, int]]]]:
    """
    使用 Needleman-Wunsch 风格的对齐算法，对齐两个句子列表。
    该版本为重构后的增强版本，允许一个台本句子匹配连续的多个 Whisper 句子（范围），以更好地处理台本与识别结果之间的差异。
    原版本存入only_align.py，供只对齐模式使用。
    使用 numpy 优化内存占用。

    返回对齐路径列表，每个元素为 (script_idx, whisper_range)，允许 None 表示插入/删除。
    whisper_range 是一个元组 (start_idx, end_idx)，表示连续的一段单词索引。
    """
    n, m = len(script_sents), len(whisper_sents)

    # 估算内存开销
    max_len = max_combine - 1  # 多词匹配预计算的长度维度
    sim_single_mem = n * m * 4 / (1024 ** 2)
    sim_multi_mem = n * m * max_len * 4 / (1024 ** 2)
    dp_mem = (n + 1) * (m + 1) * 4 / (1024 ** 2)
    op_mem = (n + 1) * (m + 1) * 1 / (1024 ** 2)
    match_len_mem = (n + 1) * (m + 1) * 2 / (1024 ** 2)
    match_start_mem = (n + 1) * (m + 1) * 4 / (1024 ** 2)
    total_mem = (sim_single_mem + sim_multi_mem + dp_mem + op_mem + match_len_mem + match_start_mem)

    logger.info(
        f"\n>>> 正在运行对齐算法(_align_sentence_lists)\n"
        f"    字幕单词数 m = {m}，台本句子数 n = {n}，max_combine K = {max_combine}\n"
        f"    时间复杂度 O(m·n·K)，空间复杂度 O(m·n·K)（预计算）+ O(m·n)（DP表）\n"
        f"    ─────── 内存估算 ───────\n"
        f"    sim_single (n×m)        : {sim_single_mem:7.2f} MB\n"
        f"    sim_multi (n×m×(K-1))   : {sim_multi_mem:7.2f} MB\n"
        f"    dp表 (int32)            : {dp_mem:7.2f} MB\n"
        f"    op表 (int8)             : {op_mem:7.2f} MB\n"
        f"    match_len表 (int16)     : {match_len_mem:7.2f} MB\n"
        f"    match_start表 (int32)   : {match_start_mem:7.2f} MB\n"
        f"    ─────────────────────────\n"
        f"    总计                    : {total_mem:7.2f} MB"
    )

    progress_start = PROGRESS_ALIGN_START
    progress_range = PROGRESS_ALIGN_END - PROGRESS_ALIGN_START

    # 估算预计算总操作数：sim_single 每个单元格一次，sim_multi 每个 (i,j,length) 一次
    total_pre_ops = n * m  # sim_single
    for i in range(n):
        for j in range(m):
            max_len_here = min(max_combine - 1, m - j - 1)  # 实际有效的长度数量
            if max_len_here > 0:
                total_pre_ops += max_len_here
    pre_ops_done = 0
    report_interval = max(1, total_pre_ops // 500)

    # 1) match_single相似度矩阵
    sim_single = np.zeros((n, m), dtype=np.int32)
    for i in range(n):
        for j in range(m):
            sim_single[i, j] = fuzz.token_set_ratio(script_sents[i], whisper_sents[j]) - similarity_offset
            pre_ops_done += 1
            if progress_queue is not None and pre_ops_done % report_interval == 0:
                ratio = pre_ops_done / total_pre_ops
                progress = int(progress_start + PRE_WEIGHT * progress_range * ratio)
                progress_queue.put(progress)

    # 2) match_multi相似度矩阵
    max_len = max_combine - 1
    # 使用-1e9作为无效值
    sim_multi = np.full((n, m, max_len), -10**9, dtype=np.int32)
    for i in range(n):
        for j in range(m):
            for length in range(2, max_combine + 1):
                start = j
                end = j + length
                if end > m:
                    break
                combined_text = ' '.join(whisper_sents[start:end])
                sim = fuzz.token_set_ratio(script_sents[i], combined_text) - similarity_offset
                sim_multi[i, j, length-2] = sim
                pre_ops_done += 1
                if progress_queue is not None and pre_ops_done % report_interval == 0:
                    ratio = pre_ops_done / total_pre_ops
                    progress = int(progress_start + PRE_WEIGHT * progress_range * ratio)
                    progress_queue.put(progress)

    # 预计算完成后，发送一次进度，进入DP阶段
    if progress_queue is not None:
        progress_queue.put(int(progress_start + PRE_WEIGHT * progress_range))

    # DP表和回溯信息表
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    op = np.zeros((n + 1, m + 1), dtype=np.int8) # 操作类型: 0=delete, 1=insert, 2=match_single, 3=match_multi

    # 对于 match_multi 记录匹配的长度 (>=2) 和起始列 j_start
    match_len = np.zeros((n + 1, m + 1), dtype=np.int16)
    match_start = np.zeros((n + 1, m + 1), dtype=np.int32)

    # 初始化边界
    for i in range(1, n + 1):
        dp[i, 0] = dp[i-1, 0] + gap_penalty
        op[i, 0] = 0
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j-1] + gap_penalty
        op[0, j] = 1

    # DP 填充阶段
    total_cells = n * m
    cells_done = 0
    report_cell_interval = max(1, total_cells // 500)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 候选分数列表
            candidates = []
            ops = []

            # 1) 匹配单个单词
            score_single = dp[i-1, j-1] + sim_single[i-1, j-1]
            candidates.append(score_single)
            ops.append(2)

            # 2) 匹配多个单词
            best_multi_score = -10**9
            best_multi_len = 0
            best_multi_start = -1
            for length in range(2, max_combine + 1):
                if j - length < 0:
                    break
                start_col = j - length
                score = dp[i-1, start_col] + sim_multi[i-1, start_col, length-2]
                if score > best_multi_score:
                    best_multi_score = score
                    best_multi_len = length
                    best_multi_start = start_col
            if best_multi_len > 0:
                candidates.append(best_multi_score)
                ops.append(3)

            # 3) delete
            score_del = dp[i-1, j] + gap_penalty
            candidates.append(score_del)
            ops.append(0)

            # 4) insert
            score_ins = dp[i, j-1] + gap_penalty
            candidates.append(score_ins)
            ops.append(1)

            best_idx = np.argmax(candidates)
            best_score = candidates[best_idx]
            best_op = ops[best_idx]

            dp[i, j] = best_score
            op[i, j] = best_op

            if best_op == 3:
                match_len[i, j] = best_multi_len
                match_start[i, j] = best_multi_start
            elif best_op == 2:
                match_len[i, j] = 1
                match_start[i, j] = j-1

            cells_done += 1
            if progress_queue is not None and cells_done % report_cell_interval == 0:
                dp_ratio = cells_done / total_cells
                progress = int(progress_start + PRE_WEIGHT * progress_range + DP_WEIGHT * progress_range * dp_ratio)
                progress_queue.put(progress)

        # 每行结束后也更新一次，保证最终行能触发完成
        if progress_queue is not None:
            dp_ratio = i / n
            progress = int(progress_start + PRE_WEIGHT * progress_range + DP_WEIGHT * progress_range * dp_ratio)
            progress_queue.put(progress)

    # 对齐完成，发送最终进度
    if progress_queue is not None:
        progress_queue.put(PROGRESS_ALIGN_END - 1)

    # 回溯
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and op[i, j] in (2, 3):
            if op[i, j] == 2:
                start_idx = match_start[i, j]
                end_idx = start_idx
                alignment.append((i-1, (start_idx, end_idx)))
                i -= 1
                j -= 1
            else:
                length = match_len[i, j]
                start_idx = match_start[i, j]
                end_idx = start_idx + length - 1
                alignment.append((i-1, (start_idx, end_idx)))
                i -= 1
                j -= length
        elif i > 0 and (j == 0 or op[i, j] == 0):
            alignment.append((i-1, None))
            i -= 1
        elif j > 0 and (i == 0 or op[i, j] == 1):
            alignment.append((None, j-1))
            j -= 1
        else:
            if i > 0:
                alignment.append((i-1, None))
                i -= 1
            elif j > 0:
                alignment.append((None, j-1))
                j -= 1

    alignment.reverse()
    logger.info(f"对齐完成，路径长度 {len(alignment)}")
    return alignment

def _transcribe_unified(
        model: WhisperModel,
        audio_path: str,
        language: str,
        beam_size: int,
        vad_filter: bool,
        vad_parameters: Dict,
        progress_queue: Optional[multiprocessing.Queue],
        verbose: Optional[bool] = True
    ) -> Tuple[List[Tuple[str, float, float]], float]:
    """
    统一转录：返回单词列表（word, start, end）和总时长，同时发送进度，并实时记录识别片段。
    """
    # 定义内部进度回调
    def progress_cb(p, eta):
        if progress_queue is not None and eta > 0:
            progress = int((p / eta) * PROGRESS_TRANSCRIBE_MAX)
            progress_queue.put(progress)
    # Stable Whisper为Faster Whisper重新封装了transcribe函数，新增了progress_callback与verbose参数。
    # 真正的transcribe函数位于stable_whisper.whisper_word_level.faster_whisper中。
    result: WhisperResult = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters if vad_filter else None,
        progress_callback=progress_cb,
        verbose=verbose
    )
    total_duration = result.ori_dict.get('duration', 0.0)
    if not isinstance(total_duration, (int, float)):
        total_duration = 0.0
    else:
        total_duration = float(total_duration)
    # 收集所有单词
    all_words = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                all_words.append((w.word.strip(), w.start, w.end))
    if total_duration <= 0 and all_words:
        total_duration = all_words[-1][2]
    if progress_queue is not None:
        progress_queue.put(PROGRESS_TRANSCRIBE_MAX)
    return all_words, total_duration

def _prepare_script(
        script_path: str,
        preprocess: bool,
        short_sentences: bool
    ) -> Tuple[str, List[str]]:
    """
    读取并分割台本，返回原始文本和句子列表。
    短句模式按标点分割，长句模式使用 pysbd。
    """
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    if preprocess:
        from just_utils import clean_script_text
        script_text = clean_script_text(script_text)
        logger.info("已对台本进行预处理（删除空行和方括号内容）。")
    logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符。")
    if short_sentences:
        script_sents = split_text_by_punctuation(script_text)
        logger.info(f"已按标点分割台本。")
    else:
        script_sents = split_text_by_pysbd(script_text)
        logger.info(f"已按pysbd分割台本。")
    return script_text, script_sents

def _build_subtitles_from_words(
        script_sents: List[str],
        all_words: List[Tuple[str, float, float]],
        gap_penalty: int, 
        similarity_offset: int, 
        default_duration: float,
        max_combine: int,
        progress_queue: Optional[multiprocessing.Queue]
    ) -> List[Tuple[str, float, float]]:
    """
    将台本句子与单词列表对齐，为每个句子分配时间戳。
    现在使用增强的对齐算法，允许一个台本句子匹配多个单词。
    """

    # 提取单词文本列表
    word_texts = [w[0] for w in all_words]

    # 对齐台本句子和单词序列
    alignment = _align_sentence_lists(script_sents, word_texts, gap_penalty, similarity_offset, max_combine, progress_queue=progress_queue)
    
    # 输出对齐日志
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
    logger.info(f"时间映射构建完成，匹配到时间的句子数: {len(time_map)} / {len(script_sents)}")

    # 生成字幕
    interpolated = interpolate_timestamps(time_map, len(script_sents), default_duration)
    logger.info(f"时间轴差值完成，生成 {len(interpolated)} 条源字幕（包含插值）。")
    result = []
    for idx, start, end in interpolated:
        text = normalize_subtitle_text(script_sents[idx])
        if not text:
            continue
        result.append((text, start, end))

    # 过滤纯标点行
    filtered = []
    for text, start, end in result:
        if not is_punctuation_only(text):
            filtered.append((text, start, end))
    filtered = normalize_subtitles(filtered)
    logger.info(f"经过过滤得到 {len(filtered)} 条字幕。")
    return filtered

def _run_whisper_task(
        audio_path: str,
        script_path: str,
        local_model_path: str,
        language: str,
        device: str,
        compute_type: str,
        result_queue: multiprocessing.Queue,
        preprocess: bool = False,
        settings: Optional[dict] = None,
        log_queue: Optional[multiprocessing.Queue] = None,
        progress_queue: Optional[multiprocessing.Queue] = None,
        short_sentences: bool = False,
        verbose: Optional[bool] = True
    ) -> None:
    """
    子进程执行的任务：加载模型、识别、对齐、生成字幕列表，并将结果放入队列。
    如果提供了 log_queue，则将日志也发送到该队列。
    如果提供了 progress_queue，则发送进度（0-100 整数）
    """
    try:
        if log_queue is not None:
            setup_subprocess_logging(log_queue)   # 子进程模式，日志发往队列
        else:
            setup_logging(console=True, file=True, clear_existing=True) # 直接运行模式，日志输出到控制台与文件
        if settings is None:
            settings = {}
        beam_size = settings.get('beam_size', 5)
        vad_filter = settings.get('vad_filter', False)
        vad_parameters = settings.get('vad_parameters', {})
        gap_penalty = settings.get('gap_penalty', -10)
        similarity_offset = settings.get('similarity_offset', 50)
        default_duration = settings.get('default_duration', 5.0)
        max_combine = settings.get('max_combine', 5)

        logger.info("正在加载模型...")
        model = stable_whisper.load_faster_whisper(local_model_path, device=device, compute_type=compute_type)
        logger.info(f'模型加载完成：{local_model_path}')

        # 获取单词列表
        logger.info("开始转录音频并获取单词列表。")
        all_words, total_duration = _transcribe_unified(model, audio_path, language, beam_size, vad_filter, vad_parameters, progress_queue, verbose)
        logger.info(f"转录完成，获取到 {len(all_words)} 个单词，总时长 {total_duration:.2f} 秒。")

        # 准备台本句子列表
        _, script_sents = _prepare_script(script_path, preprocess, short_sentences)
        logger.info(f"台本准备完成，获取到 {len(script_sents)} 个句子。")

        # 生成字幕（使用增强的对齐，允许匹配范围）
        logger.info("开始对齐台本句子与单词列表并生成字幕。")
        subtitles = _build_subtitles_from_words(
            script_sents, all_words, gap_penalty, similarity_offset, default_duration, max_combine, progress_queue
        )
        result_queue.put(('result', subtitles))
        result_queue.close()
        result_queue.join_thread()
        logger.info("处理完成，结果已放回队列，正在结束子进程。")
        time.sleep(0.5)
        if progress_queue is not None:
            progress_queue.put(PROGRESS_ALIGN_END)

    except Exception as e:
        error_msg = f"子进程发生错误：{str(e)}\n{traceback.format_exc()}"
        result_queue.put(('error', error_msg))
        logger.error(error_msg)
    finally:
        pass

def direct_it(
        audio_path: str,
        script_path: str,
        output_path: str,
        local_model_path: str,
        language: str = 'ja',
        device: str = 'cuda',
        compute_type: str = 'float16',
        preprocess: bool = False,
        config_path: str = 'config.ini',
        log_queue: Optional[multiprocessing.Queue] = None,
        progress_queue: Optional[multiprocessing.Queue] = None,
        short_sentences: bool = False,
        verbose: Optional[bool] = True
    ) -> None:
    """
    隔离 Faster Whisper / Stable Whisper 的进程，只做结果处理。
    """

    if log_queue is None:
        setup_logging(console=True, file=True, clear_existing=True)

    settings = load_config(config_path)

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_whisper_task,
        args=(
            audio_path,
            script_path,
            local_model_path,
            language,
            device,
            compute_type,
            result_queue,
            preprocess,
            settings,
            log_queue,
            progress_queue,
            short_sentences,
            verbose
        )
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
        logger.warning("子进程未及时退出，强制终止。")
        kill_process_tree(p.pid)
        p.join()

    if subtitles:
        logger.info('主进程已获取到字幕。正在保存字幕文件...')

    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)

    if progress_queue is not None:
        progress_queue.put(PROGRESS_DONE)
    logger.info("字幕文件保存完成。")

if __name__ == "__main__":
    setup_logging(console=True, file=True)
    direct_it(
        audio_path="audio.wav",                # 音频文件路径
        script_path="script.txt",               # 台本文件路径
        output_path="output.lrc",                # 输出文件路径（.srt 或 .lrc）
        local_model_path="./faster-whisper-large-v3-turbo",  # 本地模型文件夹路径
        language='ja',                           # 语言代码
        device='cuda',                           # 计算设备 'cuda' 或 'cpu'
        compute_type='float16',                   # 计算类型
        short_sentences=True,                    # 启用短句模式
        verbose=False                           # 终端输出样式
    )