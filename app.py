import os
import sys
import threading
import queue
import multiprocessing
import configparser
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import logging

from director import direct_it

def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    if 'common' not in config:
        raise RuntimeError("配置文件缺失 [common] 节，请先运行 init 命令。")
    common = config['common']
    return {
        'model': common.get('model'),
        'lang': common.get('lang'),
        'device': common.get('device'),
        'compute': common.get('compute')
    }

log_queue = multiprocessing.Queue()        # 从子进程接收日志
status_queue = queue.Queue()                # 从后台线程传递状态

def format_log_record(record):
    """将 LogRecord 对象格式化为字符串"""
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    return formatter.format(record)

def open_file_dialog(file_type):
    root = tk.Tk()
    root.withdraw()
    if file_type == 'audio':
        path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a")]
        )
    else:
        path = filedialog.askopenfilename(
            title="选择台本文件",
            filetypes=[("文本文件", "*.txt")]
        )
    root.destroy()
    return path

def processing_thread(app):
    while True:
        try:
            msg = status_queue.get(timeout=0.5)
            if msg[0] == 'start':
                (_, audio, script, name, fmt, prep) = msg
                try:
                    config = read_config()
                except Exception as e:
                    status_queue.put(('error', f"读取配置文件失败：{e}"))
                    continue

                audio_dir = os.path.dirname(audio) or '.'
                base = name if name else os.path.splitext(os.path.basename(audio))[0]
                output_path = os.path.join(audio_dir, f"{base}.{fmt}")

                try:
                    direct_it(
                        audio_path=audio,
                        script_path=script,
                        output_path=output_path,
                        local_model_path=config['model'],
                        language=config['lang'],
                        device=config['device'],
                        compute_type=config['compute'],
                        log_queue=log_queue,
                        preprocess=prep
                    )
                    status_queue.put(('success', output_path))
                except Exception as e:
                    status_queue.put(('error', str(e)))
        except queue.Empty:
            continue

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Script Director GUI")
        self.geometry("800x600")
        self.resizable(False, False)

        # 设置外观模式
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 处理状态标志
        self.is_processing = False

        # 主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # 音频文件
        self.audio_label = ctk.CTkLabel(self.main_frame, text="音频文件：")
        self.audio_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.audio_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.audio_entry.grid(row=0, column=1, padx=5, pady=5)
        self.audio_btn = ctk.CTkButton(self.main_frame, text="浏览", width=60, command=self.browse_audio)
        self.audio_btn.grid(row=0, column=2, padx=5, pady=5)

        # 台本文件
        self.script_label = ctk.CTkLabel(self.main_frame, text="台本文件：")
        self.script_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.script_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.script_entry.grid(row=1, column=1, padx=5, pady=5)
        self.script_btn = ctk.CTkButton(self.main_frame, text="浏览", width=60, command=self.browse_script)
        self.script_btn.grid(row=1, column=2, padx=5, pady=5)

        # 输出名称
        self.name_label = ctk.CTkLabel(self.main_frame, text="输出名称：")
        self.name_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.name_entry = ctk.CTkEntry(self.main_frame, width=400)
        self.name_entry.grid(row=2, column=1, padx=5, pady=5)

        # 输出格式
        self.type_label = ctk.CTkLabel(self.main_frame, text="输出格式：")
        self.type_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.type_menu = ctk.CTkOptionMenu(self.main_frame, values=["srt", "lrc"])
        self.type_menu.grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # 预处理选项
        self.preprocess_var = tk.BooleanVar()
        self.preprocess_check = ctk.CTkCheckBox(self.main_frame, text="预处理台本（删除空行和方括号标识）",
                                                variable=self.preprocess_var)
        self.preprocess_check.grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # 开始按钮
        self.start_btn = ctk.CTkButton(self.main_frame, text="开始处理", command=self.start_processing)
        self.start_btn.grid(row=5, column=0, columnspan=2, padx=5, pady=10)

        # 日志区域
        self.log_label = ctk.CTkLabel(self.main_frame, text="运行日志：")
        self.log_label.grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.log_text = ctk.CTkTextbox(self.main_frame, width=760, height=300)
        self.log_text.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

        # 启动后台线程
        self.thread = threading.Thread(target=processing_thread, args=(self,), daemon=True)
        self.thread.start()

        # 定时检查队列
        self.after(100, self.check_queues)

        # 绑定窗口关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def browse_audio(self):
        path = open_file_dialog('audio')
        if path:
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)

    def browse_script(self):
        path = open_file_dialog('script')
        if path:
            self.script_entry.delete(0, "end")
            self.script_entry.insert(0, path)

    def start_processing(self):
        audio = self.audio_entry.get()
        script = self.script_entry.get()
        name = self.name_entry.get()
        fmt = self.type_menu.get()
        prep = self.preprocess_var.get()
        if not audio or not script:
            self.append_log("错误：请填写音频文件和台本文件路径")
            return
        # 清空日志
        self.log_text.delete("1.0", "end")
        self.is_processing = True
        status_queue.put(('start', audio, script, name, fmt, prep))

    def append_log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def check_queues(self):
        # 处理日志队列
        try:
            while True:
                item = log_queue.get_nowait()
                if isinstance(item, logging.LogRecord):
                    msg = format_log_record(item)
                else:
                    msg = str(item)
                self.append_log(msg)
        except queue.Empty:
            pass
        # 处理状态队列
        try:
            msg = status_queue.get_nowait()
            if msg[0] == 'success':
                self.is_processing = False
                self.append_log(f"成功生成字幕：{msg[1]}")
                messagebox.showinfo("完成", f"字幕已生成：{msg[1]}")
            elif msg[0] == 'error':
                self.is_processing = False
                self.append_log(f"错误：{msg[1]}")
                messagebox.showerror("错误", f"处理失败：{msg[1]}")
        except queue.Empty:
            pass
        # 继续定时
        self.after(100, self.check_queues)

    def on_closing(self):
        if self.is_processing:
            result = messagebox.askyesno("确认退出", "正在处理中，强制退出可能导致字幕不完整。\n确定要退出吗？")
            if not result:
                return
        # 强制退出整个进程（包括所有子进程）
        os._exit(0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()