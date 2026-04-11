import os
import threading
import queue
import multiprocessing
import tkinter as tk
import customtkinter as ctk
import logging
from tkinter import filedialog, messagebox
from typing import Optional

from director import direct_it
from just_utils import load_config, kill_process_tree
from only_align import align_it
from main_logger import setup_logging, logger

log_queue = multiprocessing.Queue()
progress_queue = multiprocessing.Queue()
cmd_queue = queue.Queue()          # 主线程 -> 工作线程：启动命令
result_queue = queue.Queue()       # 工作线程 -> 主线程：执行结果

def format_log_record(record: logging.LogRecord) -> str:
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    return formatter.format(record)

def open_file_dialog(file_type: str, initialdir: str = '') -> str:
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
    elif file_type == 'subtitle':
        path = filedialog.askopenfilename(
            title="选择已有字幕文件",
            initialdir=initialdir,
            filetypes=[("字幕文件", "*.srt *.lrc")]
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

def processing_thread(app: 'App') -> None:
    # 日志队列的处理完全由主进程的 check_queues 负责。
    while not app.stop_event.is_set():
        try:
            msg = cmd_queue.get(timeout=0.5)
            if msg[0] == 'start':
                # 根据消息长度判断模式：长度为 11 是听写模式，长度为 12 是只对齐模式。
                if len(msg) == 11:  # 听写模式
                    (_, audio, script, name, output_format, preprocess, model_path, language, device, compute_type, short_sentences) = msg
                    subtitle_path = None
                else:  # 只对齐模式（长度为12）
                    (_, audio, script, name, output_format, preprocess, model_path, language, device, compute_type, short_sentences, subtitle_path) = msg

                try:
                    if subtitle_path:
                        # 只对齐模式
                        audio_dir = os.path.dirname(subtitle_path) or '.'
                        base = name if name else os.path.splitext(os.path.basename(subtitle_path))[0]
                        output_path = os.path.join(audio_dir, f"{base}.{output_format}")
                        align_it(
                            script_path=script,
                            subtitle_path=subtitle_path,
                            output_path=output_path,
                            output_format=output_format,
                            preprocess=preprocess,
                            short_sentences=short_sentences,
                            config_path='config.ini'
                        )
                        result_queue.put(('success', output_path))
                    else:
                        # 听写模式（原有逻辑）
                        audio_dir = os.path.dirname(audio) or '.'
                        base = name if name else os.path.splitext(os.path.basename(audio))[0]
                        output_path = os.path.join(audio_dir, f"{base}.{output_format}")

                        direct_it(
                            audio_path=audio,
                            script_path=script,
                            output_path=output_path,
                            local_model_path=model_path,
                            language=language,
                            device=device,
                            compute_type=compute_type,
                            log_queue=log_queue,
                            preprocess=preprocess,
                            progress_queue=progress_queue,
                            short_sentences=short_sentences,
                            verbose=None
                        )
                        result_queue.put(('success', output_path))
                except Exception as e:
                    result_queue.put(('error', str(e)))
        except queue.Empty:
            continue

class ToolTip:
    """简单的悬浮提示框，用于显示帮助信息"""
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event: Optional[tk.Event] = None) -> None:
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         font=("Microsoft YaHei", 10))
        label.pack()

    def leave(self, event: Optional[tk.Event] = None) -> None:
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None

class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Script Director GUI")
        self.geometry("1400x700")
        self.resizable(False, False)

        # 配置日志：输出到文件（log.log）和队列（供 GUI 显示），不输出到终端
        setup_logging(console=False, file=False, log_queue=log_queue, clear_existing=True)

        # 启动时清空日志
        with open('log.log', 'w', encoding='utf-8') as f:
            pass

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # 定义全局字体（微软雅黑，大小12）
        self.default_font = ctk.CTkFont(family="Microsoft YaHei", size=13)
        self.button_font = ctk.CTkFont(family="Microsoft YaHei", size=14, weight="bold")
        self.log_font = ctk.CTkFont(family="Microsoft YaHei", size=12)

        self.is_processing = False
        self.stop_event = threading.Event()

        # 使用 grid 布局，明确控制左右比例
        self.grid_columnconfigure(0, weight=0, minsize=500)   # 左列固定最小宽度 500
        self.grid_columnconfigure(1, weight=1)                # 右列可扩展
        self.grid_rowconfigure(0, weight=1)

        # 左侧框架
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.grid_propagate(False)   # 不自动缩放
        self.left_frame.configure(width=500)    # 明确宽度

        # 右侧框架
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.right_frame.grid_propagate(True)   # 允许扩展

        # ---------- 左侧配置区域 ----------
        # 使用网格布局，列权重设置
        self.left_frame.grid_columnconfigure(0, weight=0)   # 标签列
        self.left_frame.grid_columnconfigure(1, weight=1)   # 输入框列
        self.left_frame.grid_columnconfigure(2, weight=0)   # 按钮列

        row = 0

        # 模型路径
        self.model_label = ctk.CTkLabel(self.left_frame, text="模型路径：", anchor="e", width=100, font=self.default_font)
        self.model_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.model_entry = ctk.CTkEntry(self.left_frame, font=self.default_font)
        self.model_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.model_btn = ctk.CTkButton(self.left_frame, text="浏览", width=80, font=self.button_font, command=self.browse_model)
        self.model_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 语言代码
        self.lang_label = ctk.CTkLabel(self.left_frame, text="语言代码：", anchor="e", width=100, font=self.default_font)
        self.lang_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.lang_combo = ctk.CTkOptionMenu(self.left_frame, values=["ja", "zh", "en", "ko", "fr", "de", "ru", "es"], font=self.default_font)
        self.lang_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.lang_combo.set("ja")
        row += 1

        # 设备类型
        self.device_label = ctk.CTkLabel(self.left_frame, text="设备类型：", anchor="e", width=100, font=self.default_font)
        self.device_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.device_combo = ctk.CTkOptionMenu(self.left_frame, values=["cuda", "cpu"], font=self.default_font)
        self.device_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.device_combo.set("cuda")
        row += 1

        # 计算类型
        self.compute_label = ctk.CTkLabel(self.left_frame, text="计算类型：", anchor="e", width=100, font=self.default_font)
        self.compute_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.compute_combo = ctk.CTkOptionMenu(self.left_frame, values=["float16", "int8_float16", "int8", "float32"], font=self.default_font)
        self.compute_combo.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.compute_combo.set("float16")
        row += 1

        # 分隔线
        separator1 = ctk.CTkFrame(self.left_frame, height=2, fg_color="gray")
        separator1.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        row += 1

        # 音频文件
        self.audio_label = ctk.CTkLabel(self.left_frame, text="音频文件：", anchor="e", width=100, font=self.default_font)
        self.audio_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.audio_entry = ctk.CTkEntry(self.left_frame, font=self.default_font)
        self.audio_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.audio_btn = ctk.CTkButton(self.left_frame, text="浏览", width=80, font=self.button_font, command=self.browse_audio)
        self.audio_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 台本文件
        self.script_label = ctk.CTkLabel(self.left_frame, text="台本文件：", anchor="e", width=100, font=self.default_font)
        self.script_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.script_entry = ctk.CTkEntry(self.left_frame, font=self.default_font)
        self.script_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.script_btn = ctk.CTkButton(self.left_frame, text="浏览", width=80, font=self.button_font, command=self.browse_script)
        self.script_btn.grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 标签 Frame
        subtitle_label_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        subtitle_label_frame.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        # 问号标签
        self.help_icon = ctk.CTkLabel(subtitle_label_frame, text="❔", font=self.default_font, cursor="hand2", width=20)
        self.help_icon.pack(side="left", padx=(0, 5))
        # 文字标签
        self.subtitle_label = ctk.CTkLabel(subtitle_label_frame, text="已有字幕：", anchor="e", font=self.default_font)
        self.subtitle_label.pack(side="left")
        # 输入框和按钮
        self.subtitle_entry = ctk.CTkEntry(self.left_frame, font=self.default_font, placeholder_text="除非使用只对齐模式，该项目选填")
        self.subtitle_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")
        self.subtitle_btn = ctk.CTkButton(self.left_frame, text="浏览", width=80, font=self.button_font, command=self.browse_subtitle)
        self.subtitle_btn.grid(row=row, column=2, padx=5, pady=5)

        # 添加 Tooltip
        help_text = (
            "只对齐模式使用方法：\n"
            "1. 在「已有字幕」中选择一个 SRT 或 LRC 文件\n"
            "2. 选择台本文件\n"
            "3. 点击「开始处理」\n"
            "程序会将台本句子与已有字幕的时间轴对齐，生成新的字幕文件。\n"
            "注意：此模式无需音频文件和语音识别模型。"
        )
        ToolTip(self.help_icon, help_text)
        row += 1

        # 输出名称
        self.name_label = ctk.CTkLabel(self.left_frame, text="输出名称：", anchor="e", width=100, font=self.default_font)
        self.name_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.name_entry = ctk.CTkEntry(self.left_frame, font=self.default_font)
        self.name_entry.grid(row=row, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        row += 1

        # 输出格式
        self.type_label = ctk.CTkLabel(self.left_frame, text="输出格式：", anchor="e", width=100, font=self.default_font)
        self.type_label.grid(row=row, column=0, padx=5, pady=5, sticky="e")
        self.type_menu = ctk.CTkOptionMenu(self.left_frame, values=["srt", "lrc"], width=120, font=self.default_font)
        self.type_menu.grid(row=row, column=1, padx=5, pady=5, sticky="w")
        self.type_menu.set("srt")
        row += 1

        # 预处理选项
        self.preprocess_var = tk.BooleanVar()
        self.preprocess_check = ctk.CTkCheckBox(
            self.left_frame,
            text="预处理台本（删除空行和方括号标识）",
            variable=self.preprocess_var,
            font=self.default_font
        )
        self.preprocess_check.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="w")
        row += 1

        # 短句模式选项
        self.short_sentences_var = tk.BooleanVar()
        self.short_sentences_check = ctk.CTkCheckBox(
            self.left_frame,
            text="短句模式（按标点分割长句，生成更精确的字幕）",
            variable=self.short_sentences_var,
            font=self.default_font
        )
        self.short_sentences_check.grid(row=row, column=0, columnspan=3, padx=5, pady=10, sticky="w")
        # 绑定事件：当字幕文件变化时，动态启用/禁用短句模式复选框
        self.subtitle_entry.bind("<KeyRelease>", self.on_subtitle_change)
        self.on_subtitle_change()  # 初始状态

        row += 1
        # 开始按钮
        self.start_btn = ctk.CTkButton(self.left_frame, text="开始处理", width=150, height=35, font=self.button_font, command=self.start_processing)
        self.start_btn.grid(row=row, column=0, columnspan=3, padx=5, pady=10)

        # ---------- 右侧区域 ----------
        # 进度条区域
        self.progress_label = ctk.CTkLabel(self.right_frame, text="处理进度：", anchor="w", font=self.default_font)
        self.progress_label.pack(pady=(5, 0), padx=10, anchor="w")

        self.progress_bar = ctk.CTkProgressBar(self.right_frame)
        self.progress_bar.pack(pady=5, padx=10, fill="x")
        self.progress_bar.set(0)

        self.progress_text = ctk.CTkLabel(self.right_frame, text="0%", anchor="w", font=self.default_font)
        self.progress_text.pack(pady=(0, 10), padx=10, anchor="w")

        # 日志区域
        self.log_label = ctk.CTkLabel(self.right_frame, text="运行日志：", anchor="w", font=self.default_font)
        self.log_label.pack(pady=(10, 0), padx=10, anchor="w")

        self.log_text = ctk.CTkTextbox(self.right_frame, wrap="word", font=self.log_font)
        self.log_text.pack(pady=5, padx=10, fill="both", expand=True)

        # 启动后台线程
        self.thread = threading.Thread(target=processing_thread, args=(self,), daemon=True)
        self.thread.start()

        # 定时检查队列
        self.after(100, self.check_queues)

        # 绑定关闭事件
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.settings = load_config('config.ini')
        if self.settings:
            if self.settings.get('model'):
                self.model_entry.insert(0, self.settings['model'])
            if self.settings.get('lang'):
                self.lang_combo.set(self.settings['lang'])
            if self.settings.get('device'):
                self.device_combo.set(self.settings['device'])
            if self.settings.get('compute'):
                self.compute_combo.set(self.settings['compute'])

    def on_subtitle_change(self, event: Optional[tk.Event] = None) -> None:
        """
        当字幕文件输入框内容变化时，调整短句模式复选框的可用性。
        """
        if self.subtitle_entry.get().strip():
            # 有字幕文件，禁用短句模式复选框并添加提示
            self.short_sentences_check.configure(state="disabled")
            if self.short_sentences_var.get():
                self.short_sentences_var.set(False)
                self.append_log("注意：只对齐模式下短句模式无效，已自动禁用。")
        else:
            # 无字幕文件，恢复短句模式复选框
            self.short_sentences_check.configure(state="normal")

    def browse_audio(self) -> None:
        initial_dir = os.path.dirname(self.audio_entry.get()) if self.audio_entry.get() else ''
        path = open_file_dialog('audio', initial_dir)
        if path:
            self.audio_entry.delete(0, "end")
            self.audio_entry.insert(0, path)

    def browse_script(self) -> None:
        initial_dir = os.path.dirname(self.script_entry.get()) if self.script_entry.get() else ''
        path = open_file_dialog('script', initial_dir)
        if path:
            self.script_entry.delete(0, "end")
            self.script_entry.insert(0, path)

    def browse_subtitle(self) -> None:
        initial_dir = os.path.dirname(self.subtitle_entry.get()) if self.subtitle_entry.get() else ''
        path = open_file_dialog('subtitle', initial_dir)
        if path:
            self.subtitle_entry.delete(0, "end")
            self.subtitle_entry.insert(0, path)
            self.on_subtitle_change()

    def browse_model(self) -> None:
        initial_dir = self.model_entry.get() if self.model_entry.get() else ''
        path = open_file_dialog('model', initial_dir)
        if path:
            self.model_entry.delete(0, "end")
            self.model_entry.insert(0, path)

    def start_processing(self) -> None:
        # 运行时清空日志
        with open('log.log', 'w', encoding='utf-8') as f:
            pass
        self.log_text.delete("1.0", "end")
        audio = self.audio_entry.get()
        script = self.script_entry.get()
        subtitle = self.subtitle_entry.get()
        name = self.name_entry.get()
        output_format = self.type_menu.get()
        preprocess = self.preprocess_var.get()
        model_path = self.model_entry.get()
        language = self.lang_combo.get()
        device = self.device_combo.get()
        compute_type = self.compute_combo.get()
        short_sentences = self.short_sentences_var.get()

        if not script:
            self.append_log("错误：请填写台本文件路径")
            return
        if not subtitle and not audio:
            self.append_log("错误：请填写音频文件或已有字幕文件")
            return
        if not subtitle and not model_path:
            self.append_log("错误：请填写模型路径（听写模式需要模型）")
            return
        if not subtitle and not language:
            self.append_log("错误：请选择语言代码")
            return

        # 如果只对齐模式且启用了短句模式，在日志中警告
        if subtitle and short_sentences:
            self.append_log("警告：只对齐模式下短句模式无效，将自动禁用短句模式。")
            short_sentences = False

        self.log_text.delete("1.0", "end")
        self.progress_bar.set(0)
        self.progress_text.configure(text="0%")

        # 打印配置信息到日志
        logger.info(
f"""
========== 当前配置 ==========
[Common] model: {model_path if not subtitle else '(只对齐模式，无需模型)'}
[Common] lang: {language if not subtitle else '(只对齐模式，无需语言)'}
[Common] device: {device if not subtitle else '(只对齐模式，无需设备)'}
[Common] compute: {compute_type if not subtitle else '(只对齐模式，无需计算类型)'}
[Advanced] gap_penalty: {self.settings['gap_penalty']}
[Advanced] similarity_offset: {self.settings['similarity_offset']}
[Advanced] default_duration: {self.settings['default_duration']}
[Advanced] max_combine: {self.settings['max_combine']}
[Advanced] beam_size: {self.settings['beam_size']}
[Advanced] vad_filter: {self.settings['vad_filter']}
[Advanced] vad_parameters: {self.settings['vad_parameters']}
=============================
台本文件： {script}
{f"已有字幕文件： {subtitle}(只对齐模式)" if subtitle else f"音频文件： {audio}"}
输出名称： {name if name else '(自动生成)'}
输出格式： {output_format}
预处理台本： {preprocess}
短句模式： {short_sentences}
============================="""
        )
        self.is_processing = True
        # 根据是否有字幕文件决定命令格式
        if subtitle:
            # 只对齐模式：传递字幕文件路径
            cmd_queue.put(('start', audio, script, name, output_format, preprocess, model_path, language, device, compute_type, short_sentences, subtitle))
        else:
            # 听写模式
            cmd_queue.put(('start', audio, script, name, output_format, preprocess, model_path, language, device, compute_type, short_sentences))

    def append_log(self, msg: str) -> None:
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")

    def check_queues(self) -> None:
        # 日志队列处理
        try:
            while True:
                item = log_queue.get_nowait()
                if isinstance(item, logging.LogRecord):
                    msg = format_log_record(item)
                    self.append_log(msg)
                    try:
                        with open('log.log', 'a', encoding='utf-8') as f:
                            f.write(msg + '\n')
                    except Exception:
                        pass
                else:
                    self.append_log(str(item))
        except queue.Empty:
            pass

        # 进度队列处理
        try:
            while True:
                progress = progress_queue.get_nowait()
                self.progress_bar.set(progress / 100.0)
                self.progress_text.configure(text=f"{progress}%")
                if progress == 100:
                    self.is_processing = False
        except queue.Empty:
            pass

        # 结果队列处理
        try:
            msg = result_queue.get_nowait()
            if msg[0] == 'success':
                self.is_processing = False
                self.append_log(f"成功生成字幕：{msg[1]}")
                self.progress_bar.set(1.0)
                self.progress_text.configure(text="100%")
                messagebox.showinfo("完成", f"字幕已生成：{msg[1]}")
            elif msg[0] == 'error':
                self.is_processing = False
                self.append_log(f"错误：{msg[1]}")
                messagebox.showerror("错误", f"处理失败：{msg[1]}")
        except queue.Empty:
            pass

        self.after(100, self.check_queues)

    def on_closing(self) -> None:
        self.stop_event.set()
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