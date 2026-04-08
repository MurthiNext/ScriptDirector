from typing import List, Tuple, Dict, Optional

def interpolate_timestamps(
    time_map: Dict[int, Tuple[float, float]],
    total_sents: int,
    default_duration: float = 5.0
) -> List[Tuple[int, float, float]]:
    """
    根据已匹配句子的时间映射，为所有句子（包括未匹配的）插值生成时间。
    返回列表，每个元素为 (句子索引, 开始时间, 结束时间)
    """
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