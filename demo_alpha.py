import re, pysbd
from faster_whisper import WhisperModel
from rapidfuzz import fuzz

def split_sentences(text, language='ja'):
    """
    使用 pysbd 将文本分割成句子列表。
    :param text: 原始文本
    :param language: 语言代码（如 'ja' 日语，'zh' 中文，'en' 英文等）
    :return: 句子列表
    """
    segmenter = pysbd.Segmenter(language=language, clean=False)
    sentences = segmenter.segment(text)
    # pysbd 返回的句子可能包含尾部换行符，strip 一下
    return [s.strip() for s in sentences if s.strip()]

def align_sentence_lists(script_sents, whisper_sents, gap_penalty=-10, match_threshold=60):
    """
    使用动态规划对齐两个句子列表。
    返回对齐结果列表，每个元素为 (script_idx, whisper_idx) 或 (script_idx, None) 表示插入，或 (None, whisper_idx) 表示删除。
    """
    n, m = len(script_sents), len(whisper_sents)
    dp = [[0] * (m+1) for _ in range(n+1)]
    
    # 初始化边界（插入/删除惩罚）
    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap_penalty
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap_penalty
    
    # 填充 DP 表
    for i in range(1, n+1):
        for j in range(1, m+1):
            # 计算两个句子的相似度得分
            score = fuzz.token_set_ratio(script_sents[i-1], whisper_sents[j-1])
            # 将相似度映射到合理范围（假设 0~100，我们映射到 -50~50）
            sim_score = (score - 50)  # 可根据实际调整
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

def map_timestamps(alignment, script_sents, whisper_sents):
    """
    根据对齐结果，为每个台本句子分配时间戳。
    返回列表，每个元素为 (sentence_text, start, end)
    """
    mapped = []
    i = 0
    while i < len(alignment):
        script_idx, whisper_idx = alignment[i]
        if script_idx is not None and whisper_idx is not None:
            # 匹配上的台本句子
            sent = script_sents[script_idx]
            # 找到连续匹配的 whisper 句子（可能多个）
            start_time = whisper_sents[whisper_idx].start
            end_time = whisper_sents[whisper_idx].end
            # 向后看是否还有连续匹配（同一个台本句子对应多个 whisper 句子）
            while i+1 < len(alignment) and alignment[i+1][0] == script_idx + 1 and alignment[i+1][1] is not None:
                i += 1
                end_time = whisper_sents[alignment[i][1]].end
            mapped.append((sent, start_time, end_time))
        elif script_idx is not None and whisper_idx is None:
            # 台本句子在 whisper 中没有对应（插入），需要插值
            # 这里简单跳过，或根据上下文估算
            pass
        i += 1
    return mapped

def main(audio_path, script_path, output_srt_path, local_model_path, language='ja'):
    # 1. 加载 Whisper 模型
    model = WhisperModel(local_model_path, device="cuda", compute_type="float16")
    
    # 2. 转录获取带时间戳的 segments
    segments, info = model.transcribe(audio_path, language=language, word_timestamps=False)
    whisper_segments = list(segments)  # 每个元素有 .text, .start, .end
    
    # 3. 读取台本
    with open(script_path, 'r', encoding='utf-8') as f:
        script_text = f.read().strip()
    
    # 4. 分割句子（相同规则）
    script_sents = split_sentences(script_text, language='ja')
    whisper_sents_text = [seg.text for seg in whisper_segments]
    #whisper_sents = split_sentences(''.join(whisper_sents_text), language='ja')
    with open('1.txt','w',encoding='utf-8') as a, open('2.txt','w',encoding='utf-8') as b, open('3.txt','w',encoding='utf-8') as c:
        a.write(str(script_sents))
        #b.write(str(whisper_sents))
        c.write(str(whisper_sents_text))
    
    # 5. 序列对齐
    alignment = align_sentence_lists(script_sents, whisper_sents_text)
    
    # 6. 时间戳映射
    mapped = map_timestamps(alignment, script_sents, whisper_segments)
    
    # 7. 生成 SRT
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for idx, (text, start, end) in enumerate(mapped, 1):
            f.write(f"{idx}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{text}\n\n")
    print(f"字幕已保存至 {output_srt_path}，共 {len(mapped)} 条")

def format_time(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

if __name__ == "__main__":
    main(
        audio_path="audio.wav",
        script_path="script.txt",
        output_srt_path="output.srt",
        local_model_path="./faster-whisper-large-v3-turbo",
        language='ja'
    )