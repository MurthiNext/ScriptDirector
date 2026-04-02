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

__author__ = 'MurthiNext'
__version__ = r"1.9.9 OMG It's My AMD!!!"
__date__ = '2026/04/02'

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

def align_sentence_lists(script_sents: List[str], whisper_sents: List[str], gap_penalty: int = -10, similarity_offset: int = 50) -> List[Tuple[Optional[int], Optional[int]]]: # 主干逻辑：对齐台本与听写结果
    """
    使用 Needleman-Wunsch 风格的对齐算法，对齐两个句子列表。
    返回对齐路径列表，每个元素为 (script_idx, whisper_idx)，允许 None 表示插入/删除。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    logger.info(f"开始对齐：台本 {n} 句，当前字幕 {m} 句")

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

def split_text_by_punctuation(text: str) -> List[str]:
    text = text.replace('\r', ' ').replace('\n', ' ')
    parts = re.split(r'(?<=[。！？…、．])\s*', text)
    return [p.strip() for p in parts if p.strip()]

def is_punctuation_only(text: str) -> bool:
    """判断文本是否只包含标点符号和空白"""
    punct = set('。！？…．、，．？！；：""''（）【】《》')
    for ch in text.strip():
        if ch not in punct and not ch.isspace():
            return False
    return True

def log_alignment_mapping(script_sents: List[str], target_sents: List[str], alignment: List[Tuple[Optional[int], Optional[int]]], name_a: str = "完整句子", name_b: str = "散落的单词") -> None:
    """
    记录对齐映射关系，格式：
      完整句子 [台本编号] ↔ 单词索引 [索引列表] : 台本句子内容
          散落的单词: [索引] 单词文本, [索引] 单词文本, ...
    """
    # 建立 script_idx -> 对应的 target_idx 列表
    script_to_target = {}
    for s_idx, t_idx in alignment:
        if s_idx is not None and t_idx is not None:
            script_to_target.setdefault(s_idx, []).append(t_idx)

    logger.info(f"========== 对齐映射（{name_a} ↔ {name_b}） ==========")
    for s_idx in sorted(script_to_target.keys()):
        t_indices = sorted(script_to_target[s_idx])
        idx_str = ", ".join(str(i) for i in t_indices)
        sent_preview = script_sents[s_idx][:80] + "..." if len(script_sents[s_idx]) > 80 else script_sents[s_idx]
        logger.info(f"  {name_a} [{s_idx}] ↔ 单词索引 [{idx_str}] : {sent_preview}")
        words_detail = []
        for t_idx in t_indices:
            word_text = target_sents[t_idx][:50] + "..." if len(target_sents[t_idx]) > 50 else target_sents[t_idx]
            words_detail.append(f"[{t_idx}] {word_text}")
        logger.info(f"      {name_b}: {', '.join(words_detail)}")
    logger.info("=" * 50)

def _transcribe_unified(model, audio_path: str, language: str,
                        beam_size: int, vad_filter: bool, vad_parameters: dict,
                        progress_queue: Optional[multiprocessing.Queue]) -> Tuple[List[Tuple[str, float, float]], float]:
    """统一转录：返回单词列表（word, start, end）和总时长，同时发送进度，并实时记录识别片段"""
    logger.info("开始转录音频...")
    # 定义内部进度回调
    def progress_cb(p, eta):
        if progress_queue is not None and eta > 0:
            progress = int((p / eta) * 95)
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
        progress_queue.put(95)
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
                                progress_queue: Optional[multiprocessing.Queue]) -> List[Tuple[str, float, float]]:
    """
    将台本句子与单词列表对齐，为每个句子分配时间戳。
    由原来的map_timestamps函数改进而来，现在直接使用单词级时间戳进行对齐，插值逻辑也相应调整。
    """
    # 提取单词文本列表
    word_texts = [w[0] for w in all_words]
    # 对齐台本句子和单词序列
    alignment = align_sentence_lists(script_sents, word_texts, gap_penalty, similarity_offset)
    log_alignment_mapping(script_sents, word_texts, alignment, "台本", "单词")
    # 构建时间映射
    time_map = {}
    for s_idx, w_idx in alignment:
        if s_idx is not None and w_idx is not None:
            start = all_words[w_idx][1]
            end = all_words[w_idx][2]
            time_map[s_idx] = (start, end)

    # 生成字幕
    result = []
    total_sents = len(script_sents)
    for idx, text in enumerate(script_sents):
        text = normalize_subtitle_text(text)
        if not text:
            continue
        if idx in time_map:
            start, end = time_map[idx]
            result.append((text, start, end))
            logger.debug(f"已匹配句子 {idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")
        else:
            # 插值逻辑
            prev_idx = next((i for i in range(idx - 1, -1, -1) if i in time_map), None)
            next_idx = next((i for i in range(idx + 1, total_sents) if i in time_map), None)
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
                logger.warning(f"句子 {idx} 无任何参考时间，使用默认值 0-{default_duration} 秒")
            result.append((text, start, end))
            logger.debug(f"插值句子 {idx}: [{start:.2f}-{end:.2f}] {text[:30]}...")
        if progress_queue is not None:
            progress = 95 + (idx + 1) / total_sents * 5
            progress_queue.put(int(progress))

    # 过滤纯标点行
    filtered = []
    for text, start, end in result:
        if not is_punctuation_only(text):
            filtered.append((text, start, end))
    filtered = normalize_subtitles(filtered)
    logger.info(f"生成 {len(filtered)} 条字幕（过滤后）")
    return filtered

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
        max_combine = advanced.get('max_combine', 5)  # 暂时未使用，保留

        if device == 'cuda':
            logger.warning('警告：由于你没有使用尊贵的NVIDIA显卡，该版本运行的Stable Whisper可能会出现114514种不同的问题，一切问题请归咎到AMD身上（？）')
        logger.info(f"加载模型: {local_model_path}")
        model = stable_whisper.load_faster_whisper(local_model_path, device=device, compute_type=compute_type)

        # 统一转录，获取单词列表
        all_words, total_duration = _transcribe_unified(
            model, audio_path, language, beam_size, vad_filter, vad_parameters, progress_queue
        )

        # 准备台本句子
        _, script_sents = _prepare_script(script_path, preprocess, short_sentences)

        # 生成字幕（统一使用单词对齐）
        subtitles = _build_subtitles_from_words(
            script_sents, all_words, gap_penalty, similarity_offset, default_duration, progress_queue
        )

        result_queue.put(subtitles)
        result_queue.close()
        result_queue.join_thread()
        time.sleep(0.5)
        logger.info("处理完成，结果已放回队列。")
        if progress_queue is not None:
            progress_queue.put(100)

    except Exception as e:
        error_msg = f"子进程发生错误：{str(e)}\n{traceback.format_exc()}"
        result_queue.put(error_msg)
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
        result = result_queue.get(timeout=3600)
        if isinstance(result, str):
            logger.error(f"子进程返回错误信息: {result}")
            raise RuntimeError(f"语音识别失败: {result}")
    except Exception as e:
        if p.is_alive():
            logger.error("子进程可能卡死，正在终止...")
            p.terminate()
            p.join()
        raise RuntimeError(f"获取结果失败: {e}")

    subtitles = [r for r in result if r if r[0]]

    p.join(timeout=10)
    if p.is_alive():
        logger.warning("子进程未及时退出，强制终止")
        p.terminate()
        p.join()

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
        compute_type='float16',                   # 计算类型
        short_sentences=False                    # 启用短句模式
    )