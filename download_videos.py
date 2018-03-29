import os
import json
import string
import random
import subprocess


save_dir = '/'
with open('data/mlb-youtube-segmented.json', 'r') as f:
    data = json.load(f)
    for entry in data:
        yturl = entry['url']
        ytid = yturl.split('=')[-1]

        if os.path.exists(os.path.join(save_dir, ytid+'.mkv')):
            continue

        cmd = 'youtube-dl -f mkv '+yturl+' -o '+os.path.join(ytid+'.mkv')
        os.system(cmd)
