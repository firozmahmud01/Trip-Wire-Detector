import os
import time
import platform
import threading
import subprocess

# Global variables to track the state
_is_playing = False
_stop_event = threading.Event()
_current_process = None
_play_thread = None

# Detect the Operating System
OS_NAME = platform.system()

if OS_NAME == 'Windows':
    # Windows uses ctypes (built-in) to access the Media Control Interface
    import ctypes

def start_sound(file_path="sound.wav"):
    global _is_playing, _play_thread, _stop_event

    # If it's already playing, do nothing and let it keep playing
    if _is_playing:
        return

    if not os.path.exists(file_path):
        print(f"Error: Could not find '{file_path}' in the current directory.")
        return

    _is_playing = True
    _stop_event.clear()

    if OS_NAME == 'Windows':
        # --- WINDOWS IMPLEMENTATION ---
        # Close any previous instances, open the new one, and use the 'repeat' flag
        ctypes.windll.winmm.mciSendStringW('close MyBgSound', None, 0, None)
        ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" type mpegvideo alias MyBgSound', None, 0, None)
        ctypes.windll.winmm.mciSendStringW('play MyBgSound repeat', None, 0, None)

    elif OS_NAME == 'Darwin':
        # --- macOS IMPLEMENTATION ---
        # macOS has 'afplay' built-in. We run it in a background thread to loop it.
        _play_thread = threading.Thread(target=_unix_play_loop, args=(file_path, 'afplay'), daemon=True)
        _play_thread.start()

    elif OS_NAME == 'Linux':
        # --- LINUX IMPLEMENTATION ---
        # Linux usually has 'ffplay' or 'mpg123', but lacks a guaranteed built-in MP3 player.
        # We will try 'ffplay' (part of ffmpeg) which is standard on most desktop distros.
        _play_thread = threading.Thread(target=_unix_play_loop, args=(file_path, 'ffplay', ['-nodisp', '-autoexit']), daemon=True)
        _play_thread.start()

def stop_sound():
    global _is_playing, _stop_event, _current_process

    # If it's not playing, there's nothing to stop
    if not _is_playing:
        return

    _is_playing = False
    _stop_event.set() # Signal the loop to stop

    if OS_NAME == 'Windows':
        ctypes.windll.winmm.mciSendStringW('stop MyBgSound', None, 0, None)
        ctypes.windll.winmm.mciSendStringW('close MyBgSound', None, 0, None)
    else:
        # For macOS/Linux, kill the underlying audio process
        if _current_process:
            _current_process.terminate()
            _current_process = None

def _unix_play_loop(file_path, command, extra_args=None):
    """Background thread function for macOS and Linux to handle looping"""
    global _current_process
    if extra_args is None:
        extra_args = []
        
    while not _stop_event.is_set():
        try:
            # Start the audio process silently
            cmd_list = [command] + extra_args + [file_path]
            _current_process = subprocess.Popen(cmd_list, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for the track to finish, checking frequently if we need to stop
            while _current_process.poll() is None:
                if _stop_event.is_set():
                    _current_process.terminate()
                    break
                time.sleep(0.1)
                
        except FileNotFoundError:
            print(f"Error: The command '{command}' is not installed on your system.")
            _stop_event.set()
            break




if __name__=="__main__":
    start_sound()
    time.sleep(2)
    stop_sound()