import sys
import os
import logging
import configparser
import multiprocessing
from typing import Optional

import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, QTextEdit,
    QMessageBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from director import direct_it

logger = logging.getLogger('director')


class LogHandler(QObject, logging.Handler):
    """自定义日志处理器，将日志记录通过信号发送到主界面"""
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        self.setFormatter(formatter)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class Worker(QObject):
    """在子线程中执行耗时任务的工作类"""
    finished = pyqtSignal(object)

    def __init__(self, audio_path, script_path, output_path,
                 model_path, language, device, compute_type):
        super().__init__()
        self.audio_path = audio_path
        self.script_path = script_path
        self.output_path = output_path
        self.model_path = model_path
        self.language = language
        self.device = device
        self.compute_type = compute_type
        # 创建日志队列，用于接收子进程的实时日志
        self.log_queue = multiprocessing.Queue()

    def run(self):
        try:
            # 调用核心处理函数，传入日志队列
            direct_it(
                audio_path=self.audio_path,
                script_path=self.script_path,
                output_path=self.output_path,
                local_model_path=self.model_path,
                language=self.language,
                device=self.device,
                compute_type=self.compute_type,
                log_queue=self.log_queue
            )
            self.finished.emit(None)
        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.finished.emit(error_msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Script Director")
        self.setMinimumSize(800, 600)

        self.config_file = "config.ini"
        self.config = configparser.ConfigParser()

        self._create_ui()
        self.load_config()

        # 设置日志处理器（主进程日志）
        self.log_handler = LogHandler()
        self.log_handler.log_signal.connect(self.append_log)
        logger.addHandler(self.log_handler)

        # 用于格式化从子进程日志队列中获取的 LogRecord
        self.log_formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')

        # 工作线程相关
        self.thread = None
        self.worker = None
        self.log_timer = None

    def _create_ui(self):
        """创建界面布局"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 配置区域
        config_group = QGroupBox("配置")
        config_layout = QFormLayout(config_group)

        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("例如: D:/models/whisper-large-v3-turbo")
        config_layout.addRow("模型路径:", self.model_edit)

        self.lang_edit = QLineEdit()
        self.lang_edit.setPlaceholderText("例如: ja, zh, en")
        config_layout.addRow("语言代码:", self.lang_edit)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        config_layout.addRow("设备类型:", self.device_combo)

        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["float16", "int8"])
        config_layout.addRow("计算类型:", self.compute_combo)

        self.save_config_btn = QPushButton("保存配置")
        self.save_config_btn.clicked.connect(self.save_config)
        config_layout.addRow("", self.save_config_btn)

        main_layout.addWidget(config_group)

        # 文件选择区域
        file_group = QGroupBox("输入文件")
        file_layout = QFormLayout(file_group)

        audio_layout = QHBoxLayout()
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("选择音频文件...")
        self.audio_edit.textChanged.connect(self.update_output_path)
        audio_browse_btn = QPushButton("浏览...")
        audio_browse_btn.clicked.connect(self.browse_audio)
        audio_layout.addWidget(self.audio_edit)
        audio_layout.addWidget(audio_browse_btn)
        file_layout.addRow("音频文件:", audio_layout)

        script_layout = QHBoxLayout()
        self.script_edit = QLineEdit()
        self.script_edit.setPlaceholderText("选择台本文件 (txt)...")
        script_browse_btn = QPushButton("浏览...")
        script_browse_btn.clicked.connect(self.browse_script)
        script_layout.addWidget(self.script_edit)
        script_layout.addWidget(script_browse_btn)
        file_layout.addRow("台本文件:", script_layout)

        format_layout = QHBoxLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItems(["srt", "lrc"])
        self.format_combo.currentTextChanged.connect(self.update_output_path)
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()
        file_layout.addRow("输出格式:", format_layout)

        output_layout = QHBoxLayout()
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("自动生成或手动输入...")
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse_btn)
        file_layout.addRow("输出路径:", output_layout)

        main_layout.addWidget(file_group)

        # 运行按钮
        self.run_btn = QPushButton("运行")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self.run_task)
        main_layout.addWidget(self.run_btn)

        # 日志显示区域
        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group)

    # 配置文件操作
    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
                if 'common' in self.config:
                    common = self.config['common']
                    self.model_edit.setText(common.get('model', ''))
                    self.lang_edit.setText(common.get('lang', ''))
                    device = common.get('device', 'cuda')
                    index = self.device_combo.findText(device)
                    if index >= 0:
                        self.device_combo.setCurrentIndex(index)
                    compute = common.get('compute', 'float16')
                    index = self.compute_combo.findText(compute)
                    if index >= 0:
                        self.compute_combo.setCurrentIndex(index)
            except Exception as e:
                QMessageBox.warning(self, "警告", f"读取配置文件失败: {e}")

    def save_config(self):
        model = self.model_edit.text().strip()
        lang = self.lang_edit.text().strip()
        device = self.device_combo.currentText()
        compute = self.compute_combo.currentText()

        if not model or not lang:
            QMessageBox.warning(self, "警告", "模型路径和语言代码不能为空")
            return

        self.config['common'] = {
            'model': model,
            'lang': lang,
            'device': device,
            'compute': compute
        }
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
            QMessageBox.information(self, "成功", "配置已保存")
            logger.info(f"配置已保存到 {self.config_file}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存配置失败: {e}")

    # 文件选择
    def browse_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.wav *.mp3 *.m4a *.flac *.aac);;所有文件 (*)"
        )
        if file_path:
            self.audio_edit.setText(file_path)

    def browse_script(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择台本文件", "",
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if file_path:
            self.script_edit.setText(file_path)

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "选择输出文件", "",
            f"字幕文件 (*.{self.format_combo.currentText()});;所有文件 (*)"
        )
        if file_path:
            self.output_edit.setText(file_path)

    def update_output_path(self):
        audio = self.audio_edit.text().strip()
        if audio and not self.output_edit.isModified():
            base = os.path.splitext(audio)[0]
            ext = self.format_combo.currentText()
            self.output_edit.setText(f"{base}.{ext}")
            self.output_edit.setModified(False)

    # 日志显示
    def append_log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    # 读取子进程日志
    def read_logs(self):
        """从 worker.log_queue 中读取所有可用日志并显示"""
        if self.worker is None:
            return
        queue = self.worker.log_queue
        try:
            while True:
                record = queue.get_nowait()  # 获取 LogRecord 对象
                msg = self.log_formatter.format(record)  # 格式化为字符串
                self.append_log(msg)
        except multiprocessing.queues.Empty:
            pass

    # 运行任务
    def run_task(self):
        # 检查配置
        if not os.path.exists(self.config_file):
            reply = QMessageBox.question(
                self, "提示",
                "配置文件不存在，是否现在保存配置？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_config()
            else:
                return

        model_path = self.model_edit.text().strip()
        language = self.lang_edit.text().strip()
        device = self.device_combo.currentText()
        compute_type = self.compute_combo.currentText()

        audio = self.audio_edit.text().strip()
        script = self.script_edit.text().strip()
        output = self.output_edit.text().strip()

        if not audio or not os.path.isfile(audio):
            QMessageBox.warning(self, "警告", "请选择有效的音频文件")
            return
        if not script or not os.path.isfile(script):
            QMessageBox.warning(self, "警告", "请选择有效的台本文件")
            return
        if not output:
            QMessageBox.warning(self, "警告", "请指定输出路径")
            return
        if not model_path or not language:
            QMessageBox.warning(self, "警告", "请填写模型路径和语言代码")
            return

        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.run_btn.setEnabled(False)
        self.log_text.clear()

        # 创建工作线程
        self.thread = QThread()
        self.worker = Worker(
            audio_path=audio,
            script_path=script,
            output_path=output,
            model_path=model_path,
            language=language,
            device=device,
            compute_type=compute_type
        )
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_task_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # 启动定时器读取子进程日志
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.read_logs)
        self.log_timer.start(100)

        self.thread.start()
        logger.info("任务已启动...")

    def on_task_finished(self, error):
        """任务完成后的处理"""
        if self.log_timer:
            self.log_timer.stop()
            self.log_timer = None

        self.run_btn.setEnabled(True)
        if error is None:
            logger.info("任务成功完成！")
            QMessageBox.information(self, "完成", "字幕生成成功！")
        else:
            QMessageBox.critical(self, "错误", "处理失败，请查看日志。")
        self.thread = None
        self.worker = None

    # 窗口关闭事件
    def closeEvent(self, event):
        """重写关闭事件，确保所有子进程立即终止"""
        os._exit(0)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()