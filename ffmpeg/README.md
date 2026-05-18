# ffmpeg

处理视频文件时需要 ffmpeg 来提取音频轨。

## 安装方式

1. 从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载对应系统的版本
2. 将 `ffmpeg.exe`（Windows）或 `ffmpeg`（Linux/macOS）放入此目录
3. 或将 ffmpeg 的 bin 目录添加到系统 PATH 环境变量中

程序会按此优先级查找：`ffmpeg/` 本地目录 → 系统 PATH → 提示安装。
