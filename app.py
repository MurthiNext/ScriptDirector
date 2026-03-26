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
import signal
import psutil

from director import direct_it

def read_config():
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    if 'common' not in config:
        return None
    common = config['common']
    return {
        'model': common.get('model'),
        'lang': common.get('lang'),
        'device': common.get('device'),
        'compute': common.get('compute')
    }

log_queue = multiprocessing.Queue()
status_queue = queue.Queue()

def format_log_record(record):
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    return formatter.format(record)

def open_file_dialog(file_type, initialdir=''):
    root = tk.Tk()
    root.withdraw()
    if file_type == 'audio':
        path = filedialog.askopenfilename(
            title="选择音频文件",
            initialdir=initialdir,
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.m4a")]
        )
    elif file_type == 'script':
        path = filedialog.askopenfilename(
            title="选择台本文件",
            initialdir=initialdir,
            filetypes=[("文本文件", "*.txt")]
        )
    elif file_type == 'model':
        path = filedialog.askdirectory(
            title="选择模型文件夹",
            initialdir=initialdir
        )
    else:
        path = ''
    root.destroy()
    return path

def kill_process_tree(pid):
    """递归终止进程及其所有子进程"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        # 等待进程结束
        gone, alive = psutil.wait_procs(children, timeout=3)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass

def processing_thread(app):
    while True:
        try:
            msg = status_queue.get(timeout=0.5)
            if msg[0] == 'start':
                (_, audio, script, name, fmt, prep, model_path, language, device, compute_type) = msg
                try:
                    audio_dir = os.path.dirname(audio) or '.'
                    base = name if name else os.path.splitext(os.path.basename(audio))[0]
                    output_path = os.path.join(audio_dir, f"{base}.{fmt}")

                    direct_it(
                        audio_path=audio,
                        script_path=script,
                        output_path=output_path,
                        local_model_path=model_path,
                        language=language,
                        device=device,
                        compute_type=compute_type,
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
        self.geometry("800x700")
        self.resizable(False, False)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.is_processing = False

        # 读取配置文件默认值
        self.config_defaults = read_config()

        # 主框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # 配置网格列权重
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=0)

        row = 0

        # 模型路径
        self.model_label = ctk.CTkLabel(self.main_frame, text="模型路径：", anchor="e", width=100)
        self.model_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.model_entry = ctk.CTkEntry(self.main_frame)
        self.model_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.model_btn = ctk.CTkButton(self.main_frame, text="浏览", width=80, command=self.browse_model)
        self.model_btn.grid(row=row, column=2, padx=5, pady=5)

        row += 1

        # 语言代码
        self.lang_label = ctk.CTkLabel(self.main_frame, text="语言代码：", anchor="e", width=100)
        self.lang_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.lang_combo = ctk.CTkOptionMenu(self.main_frame, values=["ja", "zh", "en", "ko", "fr", "de", "ru", "es"])
        self.lang_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.lang_combo.set("ja")

        row += 1

        # 设备类型
        self.device_label = ctk.CTkLabel(self.main_frame, text="设备类型：", anchor="e", width=100)
        self.device_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.device_combo = ctk.CTkOptionMenu(self.main_frame, values=["cuda", "cpu"])
        self.device_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.device_combo.set("cuda")

        row += 1

        # 计算类型
        self.compute_label = ctk.CTkLabel(self.main_frame, text="计算类型：", anchor="e", width=100)
        self.compute_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.compute_combo = ctk.CTkOptionMenu(self.main_frame, values=["float16", "int8_float16", "int8", "float32"])
        self.compute_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.compute_combo.set("float16")

        row += 1

        # 分隔线
        separator = ctk.CTkFrame(self.main_frame, height=2, fg_color="gray")
        separator.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        row += 1

        # 音频文件
        self.audio_label = ctk.CTkLabel(self.main_frame, text="音频文件：", anchor="e", width=100)
        self.audio_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.audio_entry = ctk.CTkEntry(self.main_frame)
        self.audio_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.audio_btn = ctk.CTkButton(self.main_frame, text="浏览", width=80, command=self.browse_audio)
        self.audio_btn.grid(row=row, column=2, padx=5, pady=5)

        row += 1

        # 台本文件
        self.script_label = ctk.CTkLabel(self.main_frame, text="台本文件：", anchor="e", width=100)
        self.script_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.script_entry = ctk.CTkEntry(self.main_frame)
        self.script_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.script_btn = ctk.CTkButton(self.main_frame, text="浏览", width=80, command=self.browse_script)
        self.script_btn.grid(row=row, column=2, padx=5, pady=5)

        row += 1

        # 输出名称
        self.name_label = ctk.CTkLabel(self.main_frame, text="输出名称：", anchor="e", width=100)
        self.name_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.name_entry = ctk.CTkEntry(self.main_frame)
        self.name_entry.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="ew")

        row += 1

        # 输出格式
        self.type_label = ctk.CTkLabel(self.main_frame, text="输出格式：", anchor="e", width=100)
        self.type_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.type_menu = ctk.CTkOptionMenu(self.main_frame, values=["srt", "lrc"], width=120)
        self.type_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.type_menu.set("srt")

        row += 1

        # 预处理选项
        self.preprocess_var = tk.BooleanVar()
        self.preprocess_check = ctk.CTkCheckBox(
            self.main_frame,
            text="预处理台本（删除空行和方括号标识）",
            variable=self.preprocess_var
        )
        self.preprocess_check.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="w")

        row += 1

        # 开始按钮
        self.start_btn = ctk.CTkButton(self.main_frame, text="开始处理", width=150, height=35, command=self.start_processing)
        self.start_btn.grid(row=row, column=0, columnspan=3, padx=5, pady=10)

        row += 1

        # 日志区域
        self.log_label = ctk.CTkLabel(self.main_frame, text="运行日志：", anchor="w")
        self.log_label.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="w")
        row += 1

        self.log_text = ctk.CTkTextbox(self.main_frame, height=280, wrap="word")
        self.log_text.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.main_frame.grid_rowconfigure(row, weight=1)

        # 从配置文件填充默认值
        if self.config_defaults:
            if self.config_defaults.get('model'):
                self.model_entry.insert(0, self.config_defaults['model'])
            if self.config_defaults.get('lang'):
                self.lang_combo.set(self.config_defaults['lang'])
            if self.config_defaults.get('device'):
                self.device_combo.set(self.config_defaults['device'])
            if self.config_defaults.get('compute'):
                self.compute_combo.set(self.config_defaults['compute'])

        # 启动后台线程
        self.thread = threading.Thread(target=processing_thread, args=(self,), daemon=True)
        self.thread.start()

        # 定时检查队列
        self.after(100, self.check_queues)

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def browse_audio(self):
        initial_dir = os.path.dirname(self.audio_entry.get()) if self.audio_entry.get() else ''
        path = open_file_dialog('audio', initial_dir)
        if path:
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)

    def browse_script(self):
        initial_dir = os.path.dirname(self.script_entry.get()) if self.script_entry.get() else ''
        path = open_file_dialog('script', initial_dir)
        if path:
            self.script_entry.delete(0, "end")
            self.script_entry.insert(0, path)

    def browse_model(self):
        initial_dir = self.model_entry.get() if self.model_entry.get() else ''
        path = open_file_dialog('model', initial_dir)
        if path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, path)

    def start_processing(self):
        audio = self.audio_entry.get()
        script = self.script_entry.get()
        name = self.name_entry.get()
        fmt = self.type_menu.get()
        prep = self.preprocess_var.get()
        model_path = self.model_entry.get()
        language = self.lang_combo.get()
        device = self.device_combo.get()
        compute_type = self.compute_combo.get()

        if not audio or not script:
            self.append_log("错误：请填写音频文件和台本文件路径")
            return
        if not model_path:
            self.append_log("错误：请填写模型路径")
            return
        if not language:
            self.append_log("错误：请选择语言代码")
            return

        self.log_text.delete("1.0", "end")
        self.is_processing = True
        status_queue.put(('start', audio, script, name, fmt, prep, model_path, language, device, compute_type))

    def append_log(self, msg):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def check_queues(self):
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
        self.after(100, self.check_queues)

    def on_closing(self):
        if self.is_processing:
            result = messagebox.askyesno("确认退出", "正在处理中，强制退出可能导致字幕不完整。\n确定要退出吗？")
            if not result:
                return
            # 强制终止所有子进程（包括 director 启动的进程）
            for child in multiprocessing.active_children():
                child.terminate()
            for child in multiprocessing.active_children():
                child.join(timeout=1)

            try:
                kill_process_tree(os.getpid())
            except:
                pass

        os._exit(0)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = App()
    app.mainloop()