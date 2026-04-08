import os
import logging

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