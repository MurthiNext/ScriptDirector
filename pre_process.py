import re
from typing import List, Optional

def remove_brackets(text: str) -> str:
    return re.sub(r'\[[^\]]*\]|【[^】]*】', '', text)

def remove_empty_lines(lines: List[str]) -> List[str]:
    return [line for line in lines if line.strip()]

def clean_script_text(text: str) -> str:
    """
    对台本全文进行清洗：
    1. 按行分割
    2. 删除空行
    3. 删除方括号内容（支持英文 [] 和中文 【】）
    4. 删除仅含标点符号的行（可选，但保留标点）
    """
    lines = text.splitlines()
    lines = remove_empty_lines(lines)
    cleaned_lines = []
    for line in lines:
        # 删除方括号内容
        line = remove_brackets(line)
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
            f.close()
    return cleaned

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = preprocess_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(result)