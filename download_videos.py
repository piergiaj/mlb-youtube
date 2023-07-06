import os
import json
import subprocess

# 전체 영상을 받을 폴더 경로
save_dir = '/'

# yt-dlp로 유튜브 동영상 다운로드하는 함수
def download_video(url):
    command = ["yt-dlp", "-o",
               f"{save_dir}/%(title)s.%(ext)s", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", url]
    subprocess.call(command)

    
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    for entry in data:
        yturl = entry['url']
        download_video(yturl)
        
        # 다운로드 완료 시 'Download Completed !!' 라는 문구 출력
        print('Download Completed !!')
