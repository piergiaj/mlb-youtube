# MLB-YouTube Dataset

The MLB-YouTube dataset is a new, large-scale dataset consisting of 20 baseball games from the 2017 MLB post-season available on YouTube with over 42 hours of video footage. Our dataset consists of two components: segmented videos for activity recognition and continuous videos for activity classification. Our dataset is quite challenging as it is created from TV broadcast baseball games where multiple different activities share the camera angle. Further, the motion/appearance difference between the various activities is quite small.

Please see our paper ????? for more details.

EXAMPLES


# Segmented Dataset
Our segmented video dataset consists of 4,290 video clips. Each clip is annotated with the various baseball activities that occur, such as swing, hit, ball, strike, foul, etc. A video clip can contain multiple activities, so we treat this as a multi-label classification task. A full list of the activities and the number of examples of each is shown in the table below. 

| Activity | \# Examples |
|------------|-----------|
No Activity | 2983 
Ball     | 1434 
Strike   |  1799 
Swing    |  2506 
Hit      |  1391 
Foul     |  718 
In Play  |  679 
Bunt     |  24 
Hit by Pitch | 14

We additionally annotated each clip containing a pitch with the pitch type (e.g., fastball, curveball, slider, etc.) and the speed of the pitch. We also collected a set of 2,983 hard negative examples where no action occurs. These examples include views of the crowd, the field, or the players standing before or after a pitch occurred. Examples of the activities and hard negatives are shown below:

### Strike
<img src="/examples/strike1.gif?raw=true" width="250"> <img src="/examples/strike2.gif?raw=true" width="250">

### Ball
<img src="/examples/ball1.gif?raw=true" width="250"> <img src="/examples/ball2.gif?raw=true" width="250">

### Swing
<img src="/examples/swing1.gif?raw=true" width="250"> <img src="/examples/swing2.gif?raw=true" width="250">

### Hit
<img src="/examples/hit1.gif?raw=true" width="250"> <img src="/examples/hit2.gif?raw=true" width="250">

### Foul
<img src="/examples/foul1.gif?raw=true" width="250"> <img src="/examples/foul2.gif?raw=true" width="250">

### Bunt
<img src="/examples/bunt1.gif?raw=true" width="250"> <img src="/examples/bunt2.gif?raw=true" width="250">

### Hit By Pitch
<img src="/examples/hpb1.gif?raw=true" width="250"> <img src="/examples/hpb2.gif?raw=true" width="250">

# Continuous Dataset
Our continuous video dataset consists of 2,128 1-2 minute long clips from the videos. We densely annotate each frame of the clip with the baseball activities that occur.  Each continuous clip contains on average of 7.2 activities, resulting in a total of over 15,000 activity instances. Here is an example clip with instances annotated:

???




# Create the dataset
1. Download the youtube videos. Run `python download_videos.py` which relies on youtube-dl. Change the `save_dir` in the script to where you want the full videos saved.
2. To extract the segmented video clips, run `python extract_segmented_videos.py` and change `input_directory` to be the directory containing the full videos and `output_directory` to be the location to save the extracted clips.
3. To extract the continuous video clips, run `python extract_continuous_videos.py` and change `input_directory` to be the directory containing the full videos and `output_directory` to be the location to save the extracted clips.

# Baseline Experiments
...

# Requirements

- [youtube-dl](https://rg3.github.io/youtube-dl/) to download the videos
- tested with ffmpeg 2.8.11 to extract clips
