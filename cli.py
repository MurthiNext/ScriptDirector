import click
import configparser
import os, sys
import mimetypes
from director import direct_it

ask_input = input # 修改内置函数名

def exception_handler(func): # 异常处理器
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except KeyError as e:
            print('报错:',e)
            while True:
                a = ask_input('疑似配置文件出现错误，是否删除？(y/n)')
                if a in ['y', 'Y']:
                    os.remove('config.ini')
                    print('已删除配置文件。')
                    break
                elif a in ['n', 'N']:
                    print('已跳过。')
                    break
        except Exception as e:
            print('报错:',e)
    return wrapper

@click.command
@click.option('-i', '--input', type=str, help='输入文件路径(音频与台本，使用英文逗号分隔)。')
@click.option('-o', '--output', type=str, default='output.lrc', help='输出字幕路径(lrc或srt格式)。')
def main(input, output):
    if not os.path.exists('config.ini'): # 配置文件初始化
        print('未检测到配置文件，请重新配置。')
        conf = configparser.ConfigParser()
        while True:
            model = ask_input('请输入Faster Whisper本地模型路径:')
            lang = ask_input('请输入台本与音频所使用的语言代码:')
            device = ask_input('请输入设备类型(cuda/cpu):')
            compute = ask_input('请输入计算类型(float16/int8):')
            if model and lang and device and compute:
                break
            else:
                print('配置文件不能有空项，请重新输入。')
        conf['common'] = {
            'model': model,
            'lang': lang,
            'device': device,
            'compute': compute
        }
        with open('config.ini', 'w') as configfile:
            conf.write(configfile)
        sys.exit(0)

    # CLI主逻辑

    conf = configparser.ConfigParser()
    conf.read('config.ini')
    common = conf['common']
    model = common['model']
    lang = common['lang']
    device = common['device']
    compute = common['compute']
    if input:
        lst = input.split(',')
        if len(lst) != 2:
            raise ValueError('输入参数不合规。')
        audio = script = ''
        for c in lst:
            ext = c.split('.')[-1]
            filetype = mimetypes.guess_type(c)[0].split('/')[0]
            if ext == 'txt':
                script = c
            elif filetype == 'audio':
                audio = c
        # 完备的防呆机制（自豪）
        if not os.path.isfile(audio):
            raise FileNotFoundError('音频文件不存在。')
        if not os.path.isfile(script):
            raise FileNotFoundError('台本文件不存在。')
        if audio and script and output:
            # 正式运行程序
            direct_it(
                audio_path=audio,
                script_path=script,
                output_path=output,
                local_model_path=model,
                language=lang,
                device=device,
                compute_type=compute
            )

if __name__ == '__main__':
    main()