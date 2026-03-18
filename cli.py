import click
import configparser
import os
import sys
import mimetypes
from typing import Optional
from director import direct_it

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
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@cli.command(name='help', help='显示命令帮助信息。若不指定命令名称，则显示全局帮助。')
@click.argument('command_name', required=False, type=str)
def help_command(command_name: Optional[str] = None) -> None:
    """
    显示指定命令的详细用法。如果不提供命令名，则显示所有命令的摘要。
    """
    if command_name:
        if command_name not in cli.commands:
            click.echo(f"错误：未知命令 '{command_name}'")
            return
        cmd_obj = cli.commands[command_name]
        with click.Context(cmd_obj) as ctx:
            click.echo(cmd_obj.get_help(ctx))
    else:
        click.echo(cli.get_help(click.Context(cli)))

@cli.command(name='init', help='初始化配置文件（交互式）')
def init_config() -> None:
    """
    初始化配置文件 config.ini。

    你需要依次输入以下配置项：
    - Faster Whisper 本地模型路径（例如：D:/models/whisper-medium）
    - 台本与音频所使用的语言代码（例如：zh, en, ja）
    - 设备类型（cuda 或 cpu）
    - 计算类型（float16 或 int8）

    如果配置文件已存在，会询问是否覆盖。
    """
    if os.path.exists('config.ini'):
        overwrite = ask_input('配置文件已存在，是否覆盖？(y/n): ').lower()
        if overwrite != 'y':
            click.echo('已取消初始化。')
            return
    click.echo('开始配置...')
    conf = configparser.ConfigParser()
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
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        conf.write(configfile)
    click.echo('配置文件已保存。')

@cli.command(name='config', help='快速修改配置项，格式：key=value')
@click.argument('key_value', type=str)
def modify_config(key_value: str) -> None:
    """
    直接通过命令行修改配置文件中的某项配置。

    用法示例：
        python cli.py config model=/new/path/to/model
        python cli.py config lang=en

    可修改的配置项包括：model, lang, device, compute。
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
    if 'common' not in conf:
        conf['common'] = {}
    conf['common'][key] = value
    with open('config.ini', 'w', encoding='utf-8') as configfile:
        conf.write(configfile)
    click.echo(f'已更新配置项 {key} = {value}')

@exception_handler
@cli.command(name='process', help='通过台本文件与音频文件生成字幕文件')
@click.argument('input_str', type=str)
@click.option('-t', '--typing', type=click.Choice(['srt', 'lrc'], case_sensitive=False),
              default='srt', help='输出字幕格式，支持 srt 或 lrc，默认为 srt')
@click.option('-n', '--name', type=str, default=None, help='指定输出文件的名称，该项不包含扩展名')
def process_command(input_str: str, typing: str, name: str) -> None:
    """
    处理音频和台本，生成字幕文件。

    INPUT_STR 必须包含两个文件路径，用英文逗号分隔，例如：
        python cli.py process "audio.wav,script.txt"

    程序会自动区分音频文件和台本文件：
    - 扩展名为 .txt 的视为台本文件
    - 其他扩展名或 MIME 类型为 audio/ 的视为音频文件

    生成的字幕文件与音频文件同名，扩展名为 .srt 或 .lrc，保存在同一目录。

    运行此命令前必须通过 init 命令创建配置文件 config.ini。
    """
    files = [f.strip() for f in input_str.split(',')]
    if len(files) != 2:
        raise click.UsageError('输入参数必须包含两个文件路径，用逗号分隔。')

    audio_path = None
    script_path = None
    for f in files:
        if not os.path.isfile(f):
            raise click.FileError(f, f'文件不存在：{f}')
        ext = os.path.splitext(f)[1].lower()
        if ext == '.txt':
            script_path = f
        else:
            mime_type, _ = mimetypes.guess_type(f)
            if mime_type and mime_type.startswith('audio/'):
                audio_path = f
            else:
                audio_path = f

    if not script_path or not audio_path:
        raise click.UsageError('无法识别音频文件和台本文件，请确保文件扩展名正确。')

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
    output_filename = f"{audio_basename}.{typing}"
    output_path = os.path.join(audio_dir, output_filename)

    direct_it(
        audio_path=audio_path,
        script_path=script_path,
        output_path=output_path,
        local_model_path=model,
        language=lang,
        device=device,
        compute_type=compute
    )
    click.echo(f'字幕已生成：{output_path}')

if __name__ == '__main__':
    cli()