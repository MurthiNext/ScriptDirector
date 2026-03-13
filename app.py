import sys
import os
import logging
import torch
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLineEdit, QPushButton, QComboBox, QTextEdit,
    QFileDialog, QMessageBox, QStatusBar
)
from PyQt6.QtCore import QThread, pyqtSignal, QObject
from director import main as director_main

logger = logging.getLogger('director')

class QTextEditLogger(logging.Handler, QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, audio_path, script_path, output_path, model_path,
                 language, device, compute_type):
        super().__init__()
        self.audio_path = audio_path
        self.script_path = script_path
        self.output_path = output_path
        self.model_path = model_path
        self.language = language
        self.device = device
        self.compute_type = compute_type

    def run(self):
        try:
            director_main(
                self.audio_path,
                self.script_path,
                self.output_path,
                self.model_path,
                language=self.language,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            logger.error(f"处理过程中发生未捕获异常: {e}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("字幕对齐工具 (Whisper + 台本)")
        self.setMinimumSize(800, 600)

        # 设置日志处理器（在主线程）
        self.log_handler = QTextEditLogger()
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.log_handler.log_signal.connect(self.append_log)
        logger.addHandler(self.log_handler)

        # 中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 表单布局
        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)

        # 音频文件
        self.audio_edit = QLineEdit()
        self.audio_edit.setPlaceholderText("选择音频文件 (mp3, wav, m4a 等)")
        audio_browse_btn = QPushButton("浏览...")
        audio_browse_btn.clicked.connect(self.browse_audio)
        audio_layout = QHBoxLayout()
        audio_layout.addWidget(self.audio_edit)
        audio_layout.addWidget(audio_browse_btn)
        form_layout.addRow("音频文件:", audio_layout)

        # 台本文件
        self.script_edit = QLineEdit()
        self.script_edit.setPlaceholderText("选择台本文本文件 (txt)")
        script_browse_btn = QPushButton("浏览...")
        script_browse_btn.clicked.connect(self.browse_script)
        script_layout = QHBoxLayout()
        script_layout.addWidget(self.script_edit)
        script_layout.addWidget(script_browse_btn)
        form_layout.addRow("台本文件:", script_layout)

        # 输出文件
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("选择输出字幕文件 (支持 .srt 或 .lrc)")
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self.browse_output)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(output_browse_btn)
        form_layout.addRow("输出文件:", output_layout)

        # 模型路径
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("选择 Whisper 模型文件夹路径")
        model_browse_btn = QPushButton("浏览...")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_edit)
        model_layout.addWidget(model_browse_btn)
        form_layout.addRow("模型路径:", model_layout)

        # 语言选择
        self.lang_combo = QComboBox()
        languages = [
            ("日语", "ja"), ("英语", "en"), ("中文", "zh"),
            ("韩语", "ko"), ("法语", "fr"), ("德语", "de"),
            ("俄语", "ru"), ("西班牙语", "es")
        ]
        for display, code in languages:
            self.lang_combo.addItem(display, code)
        form_layout.addRow("语言:", self.lang_combo)

        # 设备选择
        self.device_combo = QComboBox()
        self.device_combo.addItem("CUDA", "cuda")
        self.device_combo.addItem("CPU", "cpu")
        form_layout.addRow("设备:", self.device_combo)

        # 计算类型
        self.compute_combo = QComboBox()
        self.compute_combo.addItem("float16", "float16")
        self.compute_combo.addItem("int8_float16", "int8_float16")
        self.compute_combo.addItem("float32", "float32")
        form_layout.addRow("计算类型:", self.compute_combo)

        layout.addWidget(form_widget)

        # 按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始处理")
        self.start_btn.clicked.connect(self.start_processing)
        btn_layout.addStretch()
        btn_layout.addWidget(self.start_btn)
        layout.addLayout(btn_layout)

        # 日志显示
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("Courier New")
        layout.addWidget(self.log_text)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

        self.thread = None
        self.worker = None

    def closeEvent(self, event):
        logger.removeHandler(self.log_handler)   # 避免窗口关闭后仍有日志尝试发送信号
        super().closeEvent(event)

    def browse_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", "",
            "音频文件 (*.mp3 *.wav *.m4a *.flac *.aac);;所有文件 (*.*)"
        )
        if file_path:
            self.audio_edit.setText(file_path)

    def browse_script(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择台本文件", "",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        if file_path:
            self.script_edit.setText(file_path)

    def browse_output(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存字幕文件", "",
            "字幕文件 (*.srt *.lrc);;SRT字幕 (*.srt);;LRC歌词 (*.lrc);;所有文件 (*.*)"
        )
        if file_path:
            if not os.path.splitext(file_path)[1]:
                file_path += ".srt"
            self.output_edit.setText(file_path)

    def browse_model(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择模型文件夹")
        if folder_path:
            self.model_edit.setText(folder_path)

    def validate_inputs(self):
        audio = self.audio_edit.text().strip()
        script = self.script_edit.text().strip()
        output = self.output_edit.text().strip()
        model = self.model_edit.text().strip()

        if not audio or not os.path.isfile(audio):
            return False, "音频文件无效"
        if not script or not os.path.isfile(script):
            return False, "台本文件无效"
        if not output:
            return False, "请指定输出文件路径"
        output_dir = os.path.dirname(output)
        if output_dir and not os.path.exists(output_dir):
            return False, f"输出目录不存在: {output_dir}"
        if not model or not os.path.isdir(model):
            return False, "模型文件夹无效"
        return True, ""

    def start_processing(self):
        valid, msg = self.validate_inputs()
        if not valid:
            QMessageBox.warning(self, "输入错误", msg)
            return

        self.start_btn.setEnabled(False)
        self.status_bar.showMessage("处理中...")
        self.log_text.clear()

        audio_path = self.audio_edit.text().strip()
        script_path = self.script_edit.text().strip()
        output_path = self.output_edit.text().strip()
        model_path = self.model_edit.text().strip()
        language = self.lang_combo.currentData()
        device = self.device_combo.currentData()
        compute_type = self.compute_combo.currentData()

        self.thread = QThread()
        self.worker = Worker(
            audio_path, script_path, output_path, model_path,
            language, device, compute_type
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.on_finished)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def append_log(self, message):
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def on_finished(self):
        self.start_btn.setEnabled(True)
        self.status_bar.showMessage("处理完成")
        QMessageBox.information(self, "完成", "字幕生成完成！")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()