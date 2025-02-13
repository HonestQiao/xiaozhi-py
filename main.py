import opuslib
import asyncio
import websockets
import pyaudio
import numpy as np
from io import BytesIO
import json
from edge_tts import Communicate
import soundfile as sf
import io
from pydub import AudioSegment
import argparse
from typing import List, Dict, Union


# 配置参数
WEBSOCKET_URI = "ws://192.168.100.11:8000"
SAMPLE_RATE = 16000  # 服务器要求16kHz采样率
CHANNELS = 1

# 添加音频设备检测
def check_audio_device():
    """检查是否有可用的音频输出设备"""
    try:
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        has_output = False

        # 检查所有设备
        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxOutputChannels') > 0:
                has_output = True
                break

        p.terminate()
        return has_output
    except Exception as e:
        print(f"[ERROR] 检查音频设备时出错: {e}")
        return False

# 全局变量记录音频设备状态
has_audio_device = check_audio_device()
if not has_audio_device:
    print("[WARN] 未检测到音频输出设备，将不会播放音频")

# 播放音频流
def play_audio_stream(audio_data):
    """直接使用 opuslib 解码 Opus 数据并用 pyaudio 播放"""
    global has_audio_device

    if not has_audio_device:
        return

    try:
        # 初始化 Opus 解码器
        decoder = opuslib.Decoder(SAMPLE_RATE, CHANNELS)

        # 初始化 PyAudio
        p = pyaudio.PyAudio()
        stream = None

        # 查找默认输出设备
        default_device_info = p.get_default_output_device_info()
        device_index = default_device_info['index']

        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=device_index,  # 指定输出设备
                frames_per_buffer=960  # 设置缓冲区大小
            )

            # Opus帧大小固定为960采样点
            frame_size = 960

            # 解码音频数据
            pcm_data = decoder.decode(audio_data, frame_size)

            # 将解码后的PCM数据写入音频流
            if stream.is_active():
                stream.write(pcm_data)

        except Exception as e:
            print(f"[ERROR] 播放音频帧失败: {e}")
            return

        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            p.terminate()

    except Exception as e:
        print(f"[ERROR] 初始化音频播放失败: {e}")
        if "no default output device available" in str(e).lower():
            has_audio_device = False
            print("[WARN] 未检测到音频设备，后续将不会尝试播放音频")

class AudioConfig:
    def __init__(self, config_json):
        data = json.loads(config_json)
        audio_params = data.get('audio_params', {})
        self.sample_rate = audio_params.get('sample_rate', 16000)
        self.channels = audio_params.get('channels', 1)
        self.frame_duration = audio_params.get('frame_duration', 60)  # 单位为毫秒
        self.format = audio_params.get('format', 'opus')
        self.session_id = data.get('session_id')

        # 计算每帧采样点数
        self.frame_size = int(self.sample_rate * (self.frame_duration / 1000))
        print(f"[INFO] 音频配置: 采样率={self.sample_rate}, 声道数={self.channels}, "
              f"帧时长={self.frame_duration}ms, 帧大小={self.frame_size}")

async def generate_tts(text: str) -> bytes:
    """使用 Edge TTS 生成语音"""
    communicate = Communicate(text, "zh-CN-XiaoxiaoNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data

async def text_to_opus_audio(text, audio_config):
    # 1. 生成 TTS 语音
    audio_data = await generate_tts(text)

    try:
        # 2. 将 MP3 数据转换为 numpy 数组
        import soundfile as sf
        import io
        from pydub import AudioSegment

        # 先用 pydub 将 MP3 转换为 WAV
        audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
        wav_data = io.BytesIO()
        audio.export(wav_data, format='wav')
        wav_data.seek(0)

        # 使用 soundfile 读取 WAV 数据
        data, samplerate = sf.read(wav_data)

        # 确保数据是 16 位整数格式
        if data.dtype != np.int16:
            data = (data * 32767).astype(np.int16)

        # 转换为字节序列
        raw_data = data.tobytes()

        print(f"[INFO] 音频参数: 采样率={samplerate}, 数据类型={data.dtype}, 长度={len(data)}")

    except Exception as e:
        print(f"[ERROR] 音频转换失败: {e}")
        return None

    # 3. 初始化Opus编码器
    encoder = opuslib.Encoder(
        audio_config.sample_rate,
        audio_config.channels,
        opuslib.APPLICATION_VOIP
    )

    # 4. 分帧编码
    frame_size = 960  # 固定使用960采样点的帧大小
    opus_frames = []

    # 按帧处理所有音频数据
    for i in range(0, len(raw_data), frame_size * 2):  # 16bit = 2bytes/sample
        chunk = raw_data[i:i + frame_size * 2]
        if len(chunk) < frame_size * 2:
            # 填充最后一帧
            chunk += b'\x00' * (frame_size * 2 - len(chunk))
        opus_frame = encoder.encode(chunk, frame_size)
        opus_frames.append(opus_frame)

    print(f"[INFO] 生成Opus音频，总帧数: {len(opus_frames)}")

    # 5. 测试解码第一帧
    decoder = opuslib.Decoder(audio_config.sample_rate, audio_config.channels)
    try:
        pcm_decoded = decoder.decode(opus_frames[0], frame_size)
        pcm_np = np.frombuffer(pcm_decoded, dtype=np.int16)
        print(f"[INFO] 成功解码测试帧，PCM 数据长度: {len(pcm_np)}")
    except Exception as e:
        print(f"[ERROR] Opus 解码失败: {e}")
        return None

    return opus_frames

class WebSocketClient:
    def __init__(self, uri: str, args=None):
        self.uri = uri
        self.websocket = None
        self.audio_config = None
        self.receive_task = None
        self._running = False
        self.args = args  # 保存命令行参数

    async def connect(self):
        """建立WebSocket连接并完成初始化"""
        self.websocket = await websockets.connect(self.uri)
        await self._send_init_config()
        response = await self.websocket.recv()
        print(f"[INFO] 收到服务器回复: {response}")
        self.audio_config = AudioConfig(response)

        # 启动后台接收任务
        self._running = True
        self.receive_task = asyncio.create_task(self._receive_loop())
        return self.audio_config

    async def _receive_loop(self):
        """后台持续接收消息的循环"""
        try:
            while self._running and self.websocket:
                try:
                    message = await self.websocket.recv()
                    await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    print("[INFO] WebSocket连接已关闭")
                    break
                except Exception as e:
                    print(f"[ERROR] 接收消息错误: {e}")
                    continue
        except Exception as e:
            print(f"[ERROR] 接收循环错误: {e}")
        finally:
            print("[INFO] 接收循环结束")

    async def _handle_message(self, message):
        """处理接收到的消息"""
        if isinstance(message, bytes):
            if has_audio_device and self.args and self.args.play_audio:  # 根据参数决定是否播放
                # 只打印较大的音频数据包
                if len(message) > 100:
                    print(f"[INFO] 收到音频数据: {len(message)} bytes")
                play_audio_stream(message)
            else:
                # 如果禁用了音频播放，只打印收到数据的信息
                print(f"[INFO] 收到音频数据: {len(message)} bytes (未播放)")
        else:
            try:
                # 尝试解析JSON消息
                data = json.loads(message)
                if isinstance(data, dict):
                    msg_type = data.get('type', 'unknown')
                    print(f"[INFO] 收到{msg_type}消息: {message}")
                else:
                    print(f"[INFO] 收到文本: {message}")
            except json.JSONDecodeError:
                print(f"[INFO] 收到文本: {message}")

    async def _send_init_config(self):
        init_config = {
            "type": "hello",
            "transport": "websocket",
            "version": 3,
            "response_mode": "auto",
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60
            }
        }
        await self.websocket.send(json.dumps(init_config))
        print("[INFO] 发送初始配置")

    async def send_text(self, text: str):
        """将文本转换为语音并发送"""
        print(f"[INFO] 发送文本: {text}")
        opus_frames = await text_to_opus_audio(text, self.audio_config)
        if opus_frames is None:
            print("[ERROR] 生成的音频数据为空")
            return

        for i, frame in enumerate(opus_frames):
            await self.websocket.send(frame)
            print(f"[INFO] 发送音频帧 {i+1}/{len(opus_frames)}")
            await asyncio.sleep(0.06)

    async def close(self):
        """关闭WebSocket连接和接收任务"""
        self._running = False
        if self.receive_task:
            try:
                self.receive_task.cancel()
                await asyncio.wait([self.receive_task], timeout=2.0)
            except Exception as e:
                print(f"[ERROR] 取消接收任务时出错: {e}")

        if self.websocket:
            await self.websocket.close()


class DialogueScenario:
    def __init__(self, messages: List[Dict[str, Union[str, float]]]):
        """
        messages格式: [
            {"text": "你好", "delay": 2.0},
            {"text": "今天天气真好", "delay": 1.5},
        ]
        """
        self.messages = messages

    async def run(self, client: WebSocketClient):
        """运行预设对话场景"""
        for msg in self.messages:
            await client.send_text(msg["text"])
            await asyncio.sleep(msg["delay"])


async def async_input(prompt: str = "") -> str:
    """异步读取用户输入"""
    # 使用线程执行器来处理阻塞的 input 调用
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def interactive_mode(client: WebSocketClient):
    """交互模式：从命令行读取用户输入并发送"""
    print("进入交互模式")
    print("- 输入 'quit' 或 'exit' 退出")
    print("- 按 Ctrl+C 退出")
    print("- 输入文字后按回车发送")
    print("-" * 50)

    while True:
        try:
            # 使用异步输入
            text = await async_input("> ")

            if text.lower() in ['quit', 'exit']:
                print("正在退出程序...")
                break
            if not text.strip():  # 跳过空输入
                continue

            await client.send_text(text)

        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，正在退出程序...")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            print("输入 'quit' 退出，或继续输入文字发送")


async def automated_test(scenario_file: str, args):
    """运行自动化测试场景"""
    with open(scenario_file, 'r', encoding='utf-8') as f:
        scenarios = json.load(f)

    client = WebSocketClient(WEBSOCKET_URI, args)
    try:
        await client.connect()
        dialogue = DialogueScenario(scenarios["messages"])
        await dialogue.run(client)

        # 等待最后的响应
        print("[INFO] 等待最后的响应...")
        await asyncio.sleep(2.0)

    finally:
        await client.close()


def parse_args():
    parser = argparse.ArgumentParser(description='WebSocket TTS 客户端')
    parser.add_argument('--mode', choices=['interactive', 'automated'],
                      default='interactive', help='运行模式')
    parser.add_argument('--scenario', type=str, help='测试场景文件路径')
    parser.add_argument('--play-audio', action='store_true',
                      help='启用音频播放 (默认: 不播放)')
    return parser.parse_args()


async def main():
    args = parse_args()
    client = WebSocketClient(WEBSOCKET_URI, args)

    try:
        await client.connect()

        if args.mode == 'interactive':
            try:
                await interactive_mode(client)
            except Exception as e:
                print(f"\n[ERROR] 发生错误: {e}")
            finally:
                print("正在关闭连接...")
        else:
            if not args.scenario:
                print("[ERROR] 自动化测试模式需要指定场景文件")
                return
            await automated_test(args.scenario, args)
    finally:
        await client.close()
        print("连接已关闭，程序结束")


if __name__ == "__main__":
    asyncio.run(main())

