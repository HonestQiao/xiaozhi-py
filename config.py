# 配置参数
WEBSOCKET_URI = "ws://192.168.1.15:8000"    # 服务器地址
SAMPLE_RATE = 16000     # 服务器要求16kHz采样率
CHANNELS = 1            # 服务器要求单声道
FRAME_DURATION = 60     # 录音块时长，单位毫秒

# 录音和静音检测参数
SOUND_THRESHOLD = 13    # 音量阈值，高于此值认为有声音
SILENCE_THRESHOLD = 12  # 静音阈值，低于此值认为无音
SILENCE_FRAMES = 15     # 静音帧数，超过此值认为静音

# 唤醒词：TODO 待实现
WAKE_WORD = "小美同学"