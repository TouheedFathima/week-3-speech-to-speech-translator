# download_audio.py
from yt_dlp import YoutubeDL
import os

# URL of the YouTube video
url = "https://www.youtube.com/watch?v=dunM6HBlskY"  


output_path = "C:/Users/WELCOME/Downloads/%(title)s.%(ext)s"


ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"  # replace if needed

ydl_opts = {
    'format': 'bestaudio/best',        # best audio quality
    'outtmpl': output_path,             # save location
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',    # convert to audio
        'preferredcodec': 'wav',        # you can change to 'mp3' if you want
        'preferredquality': '192',      # quality
    }],
    'ffmpeg_location': ffmpeg_path,     # path to ffmpeg
    'quiet': False,                     # set True to suppress output
    'noplaylist': True                  
}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])

print("Download and conversion completed!")
