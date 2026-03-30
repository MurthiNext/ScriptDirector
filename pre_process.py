import re
from typing import List, Optional

def is_bracket_line(line: str) -> bool:
    """判断整行是否仅由方括号内容（可能带空格）组成"""
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

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        result = preprocess_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
        print(result)