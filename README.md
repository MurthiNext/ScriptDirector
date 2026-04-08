# Script Director ROCm-Version
<div align=left>
   <img src="https://img.shields.io/github/v/release/MurthiNext/ScriptDirector"/>
   <img src="https://img.shields.io/github/license/MurthiNext/ScriptDirector"/>
   <img src="https://img.shields.io/github/stars/MurthiNext/ScriptDirector"/>
</div>

### 这里是Script Director的ROCm-Version分支！即**AMD特供版**，通过使用ROCm强兼CUDA实现在AMD显卡上的硬件加速！由于MurthiNext没有高性能的AMD显卡用于测试，因此这个版本可能会出现各种问题，如果你执意要使用AMD显卡加速，请尝试自行修改代码并编译。
### 此分支基于BETA-DEV，不与正式版同步更新。请确保你有足够的知识储备来解决AMD留下来的兼容性问题（因为这东西实在麻烦……）同时，**务必仔细阅读该README的所有内容**，我已为其专门设计了AMD版本的文档。

## 安装

### 依赖
- Python 3.12.10 （必须使用此版本）
- AMD ROCm 7.2.1 （若使用AMD显卡加速，必须使用此版本）
- 第三方Python库：`stable-ts`, `faster-whisper`, `psutil`, `pysbd`, `rapidfuzz`, `click`, `customtkinter`

### 在Windows10及以上的系统安装
1. 克隆或下载本项目。
2. 按照特定方法安装依赖：

   方案一：
   ```bash
   # 安装所有需要的依赖项
   # 我已为其重新编写了库来源
   pip install -r requirements.txt
   ```
   方案二：
   ```bash
   # 安装ROCm SDK
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl
   # 安装ROCm
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz
   # 安装PyTorch for ROCm 7.2.1
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
   pip install --no-cache-dir https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl
   # 安装主要依赖库，剩下的交给pip自动补全
   pip install stable-ts faster-whisper psutil pysbd rapidfuzz click customtkinter
   ```
3. 下载 Faster Whisper 模型并解压到本地目录。

### 在Linux上安装（Ubuntu 24.04）
1. 克隆或下载本项目。
2. 按照特定方法安装依赖（我尚不确定requirements.txt在Linux上是否起效）：
   ```bash
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchvision-0.24.0%2Brocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/triton-3.5.1%2Brocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
   wget https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/torchaudio-2.9.0%2Brocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl
   # 安装PyTorch for ROCm 7.2.1
   pip3 install torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp312-cp312-linux_x86_64.whl torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp312-cp312-linux_x86_64.whl torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp312-cp312-linux_x86_64.whl triton-3.5.1+rocm7.2.1.gita272dfa8-cp312-cp312-linux_x86_64.whl
   # 安装主要依赖库，剩下的交给pip自动补全
   pip3 install stable-ts faster-whisper psutil pysbd rapidfuzz click customtkinter
   ```
   理论上讲，Linux也可直接使用如下方法安装：
   ```bash
   pip3 install -r requirements.txt # 安装所有需要的依赖项
   pip3 uninstall torch torchvision triton torchaudio -y # 卸载为CUDA准备的PyTorch
   pip3 install torch torchvision triton torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 # 安装特定版本的PyTorch
   ```
   方法一是由AMD官方给出的，方法二是由PyTorch官方给出的。
3. 下载 Faster Whisper 模型并解压到本地目录。

* 你可以在[这里](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/index.html)找到AMD官方给出的技术指导，也许它会帮助你在Windows上使用ROCm部署此程序。
* 如果你是Linux用户，流程则可稍微简化一些。

#### 以下列出主流Faster Whisper模型的下载地址。
* 快上加快
   * [Faster Whisper Large V3 Turbo](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2)
* 极致精度 
   * [Faster Whisper Large V3](https://huggingface.co/Systran/faster-whisper-large-v3)
   * [Faster Whisper Large V2](https://huggingface.co/Systran/faster-whisper-large-v2)
* 性能取舍
   * [Faster Whisper Medium](https://huggingface.co/Systran/faster-whisper-medium)
   * [Faster Whisper Small](https://huggingface.co/Systran/faster-whisper-small)
   * [Faster Whisper Tiny](https://huggingface.co/Systran/faster-whisper-tiny)

以上超链接均指向[Huggingface](https://huggingface.co/)，若你身处中国大陆无法访问，可尝试使用[HF-Mirror](https://hf-mirror.com/)此镜像网站。

## 使用

具体的使用方法同main分支，AMD版本若使用ROCm计算，配置文件中`device`项依然填写`cuda`，由ROCm中间层自行兼容。