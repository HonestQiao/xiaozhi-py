# 小智同学测试工具

这是一个用于测试小智同学服务器的工具，使用 Edge TTS 进行语音合成输入的文字的音频 或者 通过麦克风录制音频，发送语音到小智同学服务器，并播放返回的语音。

## 功能特点

- 支持批量测试用例
- 自动保存生成的音频文件
- 使用websocket连接小智同学服务器

## 安装步骤

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 安装系统依赖：

- Linux:
```bash
sudo apt-get install python3-pyaudio portaudio19-dev ffmpeg
```

- MacOS:
```bash
brew install portaudio ffmpeg
```

- Windows:
  - 通常无需额外安装
  - 需要安装 [FFmpeg](https://ffmpeg.org/download.html)

## 使用方法

### 运行测试程序

```bash
# 默认运行
python main.py

# 开启声音播放
python main.py --play-audio

# 开启声音录制：启动后，输入r开始录音，不说话会自动结束播放
python main.py --use-mic
```

### 运行交互式客户端

```bash
python main.py --mode interactive
```

### 运行自动化测试场景

```bash
python main.py --mode automated --scenario scenarios/example.json
```

## 目录结构

```
.
├── README.md
├── requirements.txt
├── main.py              # 交互式客户端
├── scenarios/          # 测试场景目录
│   └── example.json   # 示例测试场景
└── test_outputs/         # 生成的音频文件目录
```

## 配置说明

### 测试场景配置

在 `scenarios` 目录下创建 JSON 文件，格式如下：

```json
{
    "name": "基础对话测试",
    "description": "测试基本的对话流程",
    "messages": [
        {
            "text": "你好，小智同学",
            "delay": 2.0
        },
        {
            "text": "今天天气怎么样？",
            "delay": 3.0
        }
    ]
}
```

### 音频参数配置

- 采样率：16kHz
- 声道数：单声道
- 位深度：16位
- 音频格式：WAV/Opus

## 注意事项

1. 音频设备
   - 确保系统有可用的音频输出设备
   - 检查音频权限设置

2. 存储空间
   - 测试音频文件会保存在 `test_outputs` 目录
   - 定期清理不需要的音频文件

## 常见问题

1. 音频播放问题
   - 检查系统音频设备
   - 确认 PyAudio 安装正确

### 调试信息

- 程序会输出详细的日志信息
- 音频文件保存在 `test_outputs` 目录
- 可以通过日志追踪测试流程

## 开发者
- 贤立
- HonestQiao

## 许可证

MIT License