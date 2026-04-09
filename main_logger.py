import logging
from multiprocessing import Queue
from logging.handlers import QueueHandler

# 单例 logger
_logger = None

def get_logger() -> logging.Logger:
    global _logger
    if _logger is None:
        _logger = logging.getLogger('director')
        _logger.setLevel(logging.INFO)
        # 初始时不添加任何处理器，由调用方配置
    return _logger

# 导出全局 logger 实例
logger = get_logger()

def setup_logging(
        console: bool = True,
        file: bool = True,
        log_queue = None,
        log_file: str = 'log.log',
        clear_existing: bool = True
    ) -> logging.Logger:
    """
    主进程日志配置：可同时输出到控制台、文件和队列。
    """
    if clear_existing:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if file:
        # 每次启动覆盖写入，可根据需要改为追加模式 'a'
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    if log_queue is not None:
        qh = QueueHandler(log_queue)
        logger.addHandler(qh)
    return logger

def setup_subprocess_logging(log_queue: Queue) -> logging.Logger:
    """
    子进程专用配置：只添加 QueueHandler，将日志发回主进程。
    """
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    qh = QueueHandler(log_queue)
    logger.addHandler(qh)
    return logger