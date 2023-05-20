import os
import json
import string
import random
import subprocess
import multiprocessing
from multiprocessing import freeze_support

# yt-dlp로 유튜브 동영상의 제목을 가져오는 함수
def get_youtube_video_title(url):
    command = ["yt-dlp", "--get-title", url]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        title = result.stdout.strip()
        return title
    else:
        print("Failed to retrieve video title")
        return None
    

def local_clip(filename, start_time, duration, output_filename, output_directory):
    end_time = start_time + duration
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'copy', '-an',
               '-threads', '1',
               '-loglevel', 'panic',
               os.path.join(output_directory,output_filename)]
    command = ' '.join(command)

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print (err.output)
        return err.output


def wrapper(clip):
    # 전체 영상이 있는 폴더 경로
    input_directory = '/'

    # 클립을 받을 폴더 경로
    output_directory = '/'
    
    duration = clip['end']-clip['start']
    video_title = get_youtube_video_title(clip['url'])
    local_clip(os.path.join(input_directory, video_title+'.mkv'), clip['start'], duration, str(clip['end'])+'.mp4', output_directory)
    return 0
    

if __name__ == '__main__':
    freeze_support()
    with open('data/mlb-youtube-segmented.json', 'r') as f:
        data = json.load(f)
        pool = multiprocessing.Pool(processes=8)
        pool.map(wrapper, [data[k] for k in data.keys() if 'swing' in data[k]['labels']])

        # 다운로드 완료 시 'Download Completed !!' 라는 문구 출력
        print('Download Completed !!')

        
