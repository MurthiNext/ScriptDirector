import click
import configparser
import os
import sys
import mimetypes
from typing import Optional
from director import direct_it
from only_align import align_only  # 新增导入

ask_input = input

def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyError as e:
            click.echo(f'报错: {e}')
            while True:
                a = ask_input('疑似配置文件出现错误，是否删除？(y/n)')
                if a.lower() == 'y':
                    os.remove('config.ini')
                    click.echo('已删除配置文件。')
                    break
                elif a.lower() == 'n':
                    click.echo('已跳过。')
                    break
        except Exception as e:
            click.echo(f'报错: {e}')
    return wrapper

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Script Director 命令行程序"""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command(name='help', short_help='显示命令帮助信息。若不指定命令名称，则显示全局帮助。')
@click.argument('command_name', required=False, type=str)
def help_command(command_name: Optional[str] = None) -> None:
    """显示指定命令的详细用法。如果不提供命令名，则显示所有命令的摘要。"""
    if command_name:
        if command_name not in cli.commands:
            click.echo(f"错误：未知命令 '{command_name}'")
            return
        cmd_obj = cli.commands[command_name]
        with click.Context(cmd_obj) as ctx:
            click.echo(cmd_obj.get_help(ctx))
    else:
        click.echo(cli.get_help(click.Context(cli)))

@cli.command(name='init', short_help='初始化配置文件（交互式）')
def init_config() -> None:
    """
    初始化配置文件 config.ini。

    \b
    你需要依次输入以下配置项：
    - Faster Whisper 本地模型路径（例如：D:/models/whisper-medium）
    - 台本与音频所使用的语言代码（例如：zh, en, ja）
    - 设备类型（cuda 或 cpu）
    - 计算类型（float16 或 int8）
    - 可选的高级配置

    如果配置文件已存在，会询问是否覆盖。
    """
    if os.path.exists('config.ini'):
        overwrite = ask_input('配置文件已存在，是否覆盖？(y/n): ').lower()
        if overwrite != 'y':
            click.echo('已取消初始化。')
            return
    click.echo('开始配置...')
    conf = configparser.ConfigParser()

    # 获取 common 节参数
    while True:
        model = ask_input('请输入Faster Whisper本地模型路径: ').strip()
        lang = ask_input('请输入台本与音频所使用的语言代码: ').strip()
        device = ask_input('请输入设备类型(cuda/cpu): ').strip()
        compute = ask_input('请输入计算类型(float16/int8): ').strip()
        if model and lang and device and compute:
            break
        click.echo('配置文件不能有空项，请重新输入。')
    conf['common'] = {
        'model': model,
        'lang': lang,
        'device': device,
        'compute': compute
    }

    # 获取 advanced 节参数（可选）
    click.echo("可选高级参数设置（直接回车使用默认值）:")
    gap_penalty = ask_input('gap_penalty (对齐惩罚值，默认 -10): ').strip()
    similarity_offset = ask_input('similarity_offset (相似度偏移，默认 50): ').strip()
    default_duration = ask_input('default_duration (默认字幕时长/秒，默认 5.0): ').strip()
    max_combine = ask_input('max_combine (最大合并片段数，默认 5): ').strip()
    beam_size = ask_input('beam_size (束搜索宽度，默认 5): ').strip()
    vad_filter = ask_input('vad_filter (启用语音活动检测，默认 False): ').strip()
    vad_parameters = ask_input('vad_parameters (VAD参数 JSON，默认 {}): ').strip()

    conf['advanced'] = {}
    if gap_penalty:
        conf['advanced']['gap_penalty'] = gap_penalty
    if similarity_offset:
        conf['advanced']['similarity_offset'] = similarity_offset
    if default_duration:
        conf['advanced']['default_duration'] = default_duration
    if max_combine:
        conf['advanced']['max_combine'] = max_combine
    if beam_size:
        conf['advanced']['beam_size'] = beam_size
    if vad_filter:
        conf['advanced']['vad_filter'] = vad_filter.lower() in ('true', '1', 'yes')
    if vad_parameters:
        conf['advanced']['vad_parameters'] = vad_parameters

    with open('config.ini', 'w', encoding='utf-8') as configfile:
        conf.write(configfile)
    click.echo('配置文件已保存。')

@cli.command(name='config', short_help='快速修改配置项，格式：key=value')
@click.argument('key_value', type=str)
def modify_config(key_value: str) -> None:
    """
    直接通过命令行修改配置文件中的某项配置。

    \b
    用法示例：
        python cli.py config model=/new/path/to/model
        python cli.py config lang=en

    可修改的配置项有common与advanced节，详见已经初始化的config.ini。
    """
    if not os.path.exists('config.ini'):
        click.echo('错误：配置文件不存在，请先运行 init 命令。')
        sys.exit(1)
    if '=' not in key_value:
        click.echo('错误：参数格式应为 key=value')
        sys.exit(1)
    key, value = key_value.split('=', 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        click.echo('错误：键和值不能为空')
        sys.exit(1)

    conf = configparser.ConfigParser()
    conf.read('config.ini', encoding='utf-8')

    # 判断属于哪个节（默认 common）
    advanced_keys = ['gap_penalty', 'similarity_offset', 'default_duration', 'max_combine', 'beam_size', 'vad_filter', 'vad_parameters']
    section = 'advanced' if key in advanced_keys else 'common'
    if section not in conf:
        conf[section] = {}
    conf[section][key] = value

    with open('config.ini', 'w', encoding='utf-8') as configfile:
        conf.write(configfile)
    click.echo(f'已更新配置项 {section}.{key} = {value}')

@exception_handler
@cli.command(name='process', short_help='通过台本文件与音频文件生成字幕文件')
@click.argument('input_str', type=str)
@click.option('-t', '--type', type=click.Choice(['srt', 'lrc'], case_sensitive=False),
              default='srt', help='输出字幕格式，支持 srt 或 lrc，默认为 srt')
@click.option('-n', '--name', type=str, default=None, help='指定输出文件的名称，该项不包含扩展名')
@click.option('-p', '--preprocess', is_flag=True, default=False,
              help='预处理台本，删除空行和方括号内容（如角色标识）')
@click.option('-s', '--shorter', is_flag=True, default=False,
              help='启用短句模式（按标点分割长句，生成更精确的字幕）')
def process_command(input_str: str, type: str, name: str, preprocess: bool, shorter: bool) -> None:
    """
    处理音频和台本，生成字幕文件。

    \b
    INPUT_STR 必须包含两个文件路径，用英文逗号分隔，例如：
        python cli.py process "audio.wav,script.txt"
        或
        python cli.py process "script.txt,subtitles.srt"

    \b
    程序会自动识别文件类型：
    - 扩展名为 .txt 的视为台本文件
    - 扩展名为 .srt 或 .lrc 的视为已有字幕文件（启用只对齐模式）
    - 其他扩展名或 MIME 类型为 audio/ 的视为音频文件

    生成的字幕文件与音频文件同名，扩展名为 .srt 或 .lrc，保存在同一目录。
    如果输入的是字幕文件（只对齐模式），则输出文件默认与字幕文件同名。

    运行此命令前必须通过 init 命令创建配置文件 config.ini。

    \b
    如果指定 -p 或 --preprocess，则会对台本进行清洗：
    - 删除空行
    - 删除方括号内的内容（例如 [人名]、[动作说明]）
    - 去除多余空格

    如果指定 -s 或 --shorter，则启用短句模式，按标点分割长句，生成更精确的字幕。
    注意：只对齐模式下短句模式无效，程序会发出警告并忽略该选项。
    """
    files = [f.strip() for f in input_str.split(',')]
    if len(files) != 2:
        raise click.UsageError('输入参数必须包含两个文件路径，用逗号分隔。')

    # 识别文件类型
    script_path = None
    audio_path = None
    subtitle_path = None
    for f in files:
        if not os.path.isfile(f):
            raise click.FileError(f, f'文件不存在：{f}')
        ext = os.path.splitext(f)[1].lower()
        if ext == '.txt':
            script_path = f
        elif ext in ('.srt', '.lrc'):
            subtitle_path = f
        else:
            mime_type, _ = mimetypes.guess_type(f)
            if mime_type and mime_type.startswith('audio/'):
                audio_path = f
            else:
                audio_path = f

    if not script_path:
        raise click.UsageError('未找到台本文件（.txt）')

    # 只对齐模式：提供了字幕文件
    if subtitle_path:
        click.echo("检测到已有字幕文件，启用只对齐模式（不进行语音识别）")
        # 如果用户指定了短句模式，发出警告
        if shorter:
            click.echo("警告：只对齐模式下短句模式无效，将忽略 -s/--shorter 选项。")
        # 确定输出路径
        output_dir = os.path.dirname(subtitle_path) or '.'
        if name:
            base = name
        else:
            base = os.path.splitext(os.path.basename(subtitle_path))[0]
        output_path = os.path.join(output_dir, f"{base}.{type}")
        # 调用 only_align.align_only
        align_only(
            script_path=script_path,
            subtitle_path=subtitle_path,
            output_path=output_path,
            output_format=type,
            preprocess=preprocess,
            short_sentences=shorter,   # 传入但函数内部会忽略
            config_path='config.ini'
        )
        click.echo(f'字幕已生成：{output_path}')
        return

    # 否则走原有流程（音频+台本）
    if not audio_path:
        raise click.UsageError('未找到音频文件或已有字幕文件')
    
    if not os.path.exists('config.ini'):
        raise click.ClickException('配置文件不存在，请先运行 init 命令。')

    conf = configparser.ConfigParser()
    conf.read('config.ini', encoding='utf-8')
    if 'common' not in conf:
        raise KeyError('配置文件中缺少 [common] 节')
    common = conf['common']
    model = common.get('model')
    lang = common.get('lang')
    device = common.get('device')
    compute = common.get('compute')
    if not all([model, lang, device, compute]):
        raise ValueError('配置文件不完整，请重新运行 init 或检查 config.ini')

    audio_dir = os.path.dirname(audio_path) or '.'
    if name:
        audio_basename = name
    else:
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_filename = f"{audio_basename}.{type}"
    output_path = os.path.join(audio_dir, output_filename)

    direct_it(
        audio_path=audio_path,
        script_path=script_path,
        output_path=output_path,
        local_model_path=model,
        language=lang,
        device=device,
        compute_type=compute,
        preprocess=preprocess,
        short_sentences=shorter
    )
    click.echo(f'字幕已生成：{output_path}')

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    cli()