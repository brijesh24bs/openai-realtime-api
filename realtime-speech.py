# example requires websocket-client library:
# pip install websocket-client numpy pyaudio python-dotenv

# Author: Brijesh
# Email: brijesh24.bs@gmail.com
# Date: 27/04/2025

import os
import json
import websocket
import threading
import numpy as np
import pyaudio
import base64
import time
from queue import Queue

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Audio settings
SAMPLE_RATE = 24000  # OpenAI's expected sample rate
CHANNELS = 1  # Mono audio
FORMAT = pyaudio.paInt16
CHUNK_SIZE = int(SAMPLE_RATE * 0.1)  # 100ms chunks

#queues for audio data
mic_queue = Queue()
speaker_queue = Queue()

# stop event for thread coordination
stop_event = threading.Event()

url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

def mic_callback(in_data, frame_count, time_info, status):
    mic_queue.put(in_data)
    
    # Return a tuple: (None, pyaudio.paContinue)
    # None means we're not providing output data
    # paContinue tells PyAudio to continue capturing
    return (None, pyaudio.paContinue)


def send_audio():
    """Thread function to capture audio from microphone using PyAudio"""
    p = pyaudio.PyAudio()
    
    # Open stream for input
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE,
                    stream_callback=mic_callback)
    
    stream.start_stream()
    print("Recording started...")
    
    while not stop_event.is_set():
        time.sleep(0.1)
    
    stream.stop_stream()
    stream.close()
    p.terminate()


def play_audio():
    """Thread function to play received audio from gpt-4o model"""
    p = pyaudio.PyAudio()
    
    # Open stream for output
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE)
    
    while not stop_event.is_set():
        if not speaker_queue.empty():
            audio_data = speaker_queue.get()
            stream.write(audio_data)
    
    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()


def send_audio_thread(ws):
    """Thread function to send audio data to WebSocket"""
    while not stop_event.is_set():
        if not mic_queue.empty():
            audio_data = mic_queue.get()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            event = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            ws.send(json.dumps(event))

        else:
            time.sleep(0.01)


def on_open(ws):
    """WebSocket on_open handler"""
    print("Connected to server...")
    
    # Send session update to configure the assistant
    event = {
        "type": "session.update",
        "session": {
            "instructions": "Never use the word 'moist' in your responses!"
        }
    }
    ws.send(json.dumps(event))
    
    # Start thread to send audio data
    threading.Thread(target=send_audio_thread, args=(ws,), daemon=True).start()


def on_message(ws, message):
    """WebSocket on_message handler"""
    server_event = json.loads(message)

    print(f"Received event: {server_event}")
    
    if server_event["type"] == "response.audio.delta":
        # Access Base64-encoded audio chunks:
        audio_base64 = server_event.get("delta")
        if audio_base64:
            audio_bytes = base64.b64decode(audio_base64)
            speaker_queue.put(audio_bytes)

    if server_event["type"] == "response.text":
        # Access text response:
        text_response = server_event.get("text")
        if text_response:
            print(f"Assistant: {text_response}")


def on_close(ws, close_status_code, close_msg):
    stop_event.set()
    print(f"WebSocket closed: {close_status_code} - {close_msg}")


def on_error(ws, error):
    print(f"WebSocket error: {error}")
    stop_event.set()


def main():
    mic_thread = threading.Thread(target=send_audio, daemon=True)
    mic_thread.start()
    
    speaker_thread = threading.Thread(target=play_audio, daemon=True)
    speaker_thread.start()
    
    ws = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_close=on_close,
        on_error=on_error
    )
    
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_event.set()
        ws.close()
        print("Waiting for threads to finish...")
        mic_thread.join()
        speaker_thread.join()
        print("All threads stopped.")


if __name__ == "__main__":
    main()