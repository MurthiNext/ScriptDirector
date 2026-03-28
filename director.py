import os
import logging
import pysbd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz
import multiprocessing
import traceback
import time
from typing import List, Tuple, Optional, Union, Any
from logging.handlers import QueueHandler
import configparser
import json

__author__ = 'MurthiNext'
__version__ = '1.2.5 Release'
__date__ = '2026/03/28'

if os.path.isfile('log.log'):
    with open('log.log','w',encoding='utf-8') as wf:
        wf.write('')
        wf.close()
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

def exception_handler(func): # 异常处理器
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(e)
    return wrapper

def load_advanced_config(config_path='config.ini'):
    """读取 [advanced] 节的配置，返回字典，未设置的项使用默认值"""
    config = configparser.ConfigParser()
    defaults = {
        'gap_penalty': '-10',
        'similarity_offset': '50',
        'default_duration': '5.0',
        'max_combine': '5',
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
    advanced = {
        'gap_penalty': int(defaults['gap_penalty']),
        'similarity_offset': int(defaults['similarity_offset']),
        'default_duration': float(defaults['default_duration']),
        'max_combine': int(defaults['max_combine']),
        'beam_size': int(defaults['beam_size']),
        'vad_filter': defaults['vad_filter'].lower() in ('true', '1', 'yes'),
        'vad_parameters': json.loads(defaults['vad_parameters']),
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
    with open(output_path, 'w', encoding='utf-8') as f:
        for text, start, _ in subtitles:
            f.write(f"{format_time_lrc(start)} {text}\n")
        f.close()
    logger.info(f"已保存 LRC 歌词到 {output_path}")

def split_sentences_pysbd(text: str, language: str = 'ja') -> List[str]:
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    result = [s.strip() for s in sentences if s.strip()]
    logger.info(f"台本分割为 {len(result)} 个句子")
    for i, sent in enumerate(result, 1):
        logger.debug(f"句子 {i}: {sent[:50]}..." if len(sent) > 50 else f"句子 {i}: {sent}")
    return result

@exception_handler
def align_sentence_lists(script_sents: List[str], whisper_sents: List[str], gap_penalty: int = -10, similarity_offset: int = 50) -> List[Tuple[Optional[int], Optional[int]]]: # 主干逻辑：对齐台本与听写结果
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
            sim_score = fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - similarity_offset
            match_score = dp[i-1][j-1] + sim_score
            delete = dp[i-1][j] + gap_penalty
            insert = dp[i][j-1] + gap_penalty
            dp[i][j] = max(match_score, delete, insert)

    # 回溯
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1]) - similarity_offset):
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
def map_timestamps(alignment: List[Tuple[Optional[int], Optional[int]]], script_sents: List[str], whisper_segments: List[Any], default_duration: float = 5.0, max_combine: int = 5, progress_queue: Optional[multiprocessing.Queue] = None) -> List[Tuple[str, float, float]]: # 主干逻辑：对齐时间轴
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
        # 限制合并的片段数量（max_combine）
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

def _run_whisper_task(audio_path: str, script_path: str, output_path: str,
                      local_model_path: str, language: str, device: str,
                      compute_type: str, result_queue: multiprocessing.Queue,
                      preprocess: bool = False,
                      advanced: Optional[dict] = None,
                      log_queue: Optional[multiprocessing.Queue] = None,
                      progress_queue: Optional[multiprocessing.Queue] = None) -> None:
    """
    子进程执行的任务：加载模型、识别、对齐、生成字幕列表，并将结果放入队列。
    如果提供了 log_queue，则将日志也发送到该队列。
    如果提供了 progress_queue，则发送进度（0-100 整数）
    """
    try:
        # 如果提供了日志队列，则添加 QueueHandler
        if log_queue is not None:
            queue_handler = QueueHandler(log_queue)
            logger.addHandler(queue_handler)

        logger.info(f"加载模型: {local_model_path}")
        beam_size = advanced.get('beam_size', 5) if advanced else 5
        vad_filter = advanced.get('vad_filter', False) if advanced else False
        vad_parameters = advanced.get('vad_parameters', {}) if advanced else {}
        gap_penalty = advanced.get('gap_penalty', -10) if advanced else -10
        similarity_offset = advanced.get('similarity_offset', 50) if advanced else 50
        default_duration = advanced.get('default_duration', 5.0) if advanced else 5.0
        max_combine = advanced.get('max_combine', 5) if advanced else 5
        logger.info(f"使用高级参数: gap_penalty={gap_penalty}, similarity_offset={similarity_offset}, "
                    f"default_duration={default_duration}, max_combine={max_combine}, beam_size={beam_size}, "
                    f"vad_filter={vad_filter}, vad_parameters={vad_parameters}")
        model = WhisperModel(local_model_path, device=device, compute_type=compute_type)

        logger.info("开始转录音频...")
        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=False,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_parameters if vad_filter else None
        )

        total_duration = info.duration  # 总时长（秒）
        whisper_segments = []
        for idx, seg in enumerate(segments):
            whisper_segments.append(seg)
            # 发送识别进度 (0-95%)
            if progress_queue is not None and total_duration > 0:
                progress = int((seg.end / total_duration) * 95)
                progress_queue.put(progress)
            logger.info(f"识别片段 {idx}\t{format_time_lrc(seg.start)}-{format_time_lrc(seg.end)} | {seg.text}")

        logger.info(f"识别完成，共 {len(whisper_segments)} 个片段")
        if progress_queue is not None:
            progress_queue.put(95)  # 识别阶段完成，进入对齐阶段

        # 读取台本
        with open(script_path, 'r', encoding='utf-8') as f:
            script_text = f.read().strip()

        # 如果启用预处理，则清洗台本内容
        if preprocess:
            from pre_process import clean_script_text
            script_text = clean_script_text(script_text)
            logger.info("已对台本进行预处理（删除空行和方括号内容）")

        logger.info(f"台本文件读取完成，长度 {len(script_text)} 字符")

        script_sents = split_sentences_pysbd(script_text, language=language)
        whisper_sents = [seg.text for seg in whisper_segments]
        alignment = align_sentence_lists(script_sents, whisper_sents, gap_penalty, similarity_offset)
        # 传递 progress_queue 以发送对齐进度
        subtitles = map_timestamps(alignment, script_sents, whisper_segments, default_duration, max_combine, progress_queue)

        result_queue.put(subtitles)
        # 确保数据被发送到管道
        result_queue.close()
        result_queue.join_thread()
        # 缓冲时间
        time.sleep(0.5)
        logger.info("处理完成，结果已放回队列。")
        # 发送完成信号（进度100%）
        if progress_queue is not None:
            progress_queue.put(100)
    except Exception as e:
        error_msg = f"子进程发生错误：{str(e)}\n{traceback.format_exc()}"
        result_queue.put(error_msg)
        logger.error(error_msg)
    finally:
        # 确保子进程退出
        pass

@exception_handler
def direct_it(audio_path: str, script_path: str, output_path: str,
              local_model_path: str, language: str = 'ja',
              device: str = 'cuda', compute_type: str = 'float16',
              preprocess: bool = False,
              config_path: str = 'config.ini',
              log_queue: Optional[multiprocessing.Queue] = None,
              progress_queue: Optional[multiprocessing.Queue] = None) -> None:
    """
    多进程隔离Faster Whisper，直接给这玩意丢进子进程里。
    新增 log_queue 参数，用于接收子进程的实时日志。
    新增 preprocess 参数，若为 True 则对台本进行清洗（删除空行、方括号内容等）。
    新增 progress_queue 参数，用于接收子进程的进度（0-100 整数）。
    """
    advanced = load_advanced_config(config_path)

    result_queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_whisper_task,
        args=(audio_path, script_path, output_path, local_model_path,
                language, device, compute_type, result_queue,
                preprocess, advanced, log_queue, progress_queue)
    )
    p.start()
    logger.info("已启动子进程进行语音识别...")

    subtitles = None
    try:
        result = result_queue.get(timeout=3600)
        if isinstance(result, str):
            logger.error(f"子进程返回错误信息: {result}")
            raise RuntimeError(f"语音识别失败: {result}")
        subtitles = result
    except Exception as e:
        if p.is_alive():
            logger.error("子进程可能卡死，正在终止...")
            p.terminate()
            p.join()
        raise RuntimeError(f"获取结果失败: {e}")

    p.join(timeout=10)
    if p.is_alive():
        logger.warning("子进程未及时退出，强制终止")
        p.terminate()
        p.join()

    # 保存字幕
    ext = os.path.splitext(output_path)[1].lower()
    if ext == '.lrc':
        save_lrc(subtitles, output_path)
    else:
        save_srt(subtitles, output_path)

    logger.info("字幕文件保存完成。")

if __name__ == "__main__":
    direct_it(
        audio_path="audio.wav",                # 音频文件路径
        script_path="script.txt",               # 台本文件路径
        output_path="output.lrc",                # 输出文件路径（.srt 或 .lrc）
        local_model_path="./faster-whisper-large-v3-turbo",  # 本地模型文件夹路径
        language='ja',                           # 语言代码
        device='cuda',                           # 计算设备 'cuda' 或 'cpu'
        compute_type='float16'                    # 计算类型
    )