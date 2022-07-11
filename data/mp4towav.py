"""
本文件基于使用ffmpeg将
MELD数据集中的mp4文件转换为 wav文件
采样率先不管，等后面构建dataset时候再处理
"""
import os
from librosa.util import find_files
from ffmpy3 import FFmpeg
import re
import subprocess
import pathlib

# mp4转wav

train_path = r"./MELD/train_splits"  #放mp4的文件夹
train_outwav_path = r"./MELD/train_splits/wav/"  #输出wav的文件夹
dev_path = r"./MELD/dev_splits_complete"
dev_outwav_path = r"./MELD/dev_splits_complete/wav/"
test_path = r"./MELD/output_repeated_splits_test"
test_outwav_path = r"./MELD/output_repeated_splits_test/wav/"


if not pathlib.Path(train_outwav_path).exists():
    os.mkdir(train_outwav_path)
if not pathlib.Path(dev_outwav_path).exists():
    os.mkdir(dev_outwav_path)
if not pathlib.Path(test_outwav_path).exists():
    os.mkdir(test_outwav_path)



dic = {"train":(train_path,train_outwav_path),
       "dev":(dev_path,dev_outwav_path),
       "test":(test_path,test_outwav_path)}


for key,value in dic.items():

    sh_path = "./prepareAudio_"+key+".sh"



    filepath = value[0]# 存放mp4视频的path
    output_wav_dir = value[1]  #输出wav的path

    mp4s = find_files(filepath,ext="mp4")    #存放了mp4的绝对路径名
    abspath = "/data/qzy/meld-sentiment-analysis/data"
    mp4s = [mp4.replace(abspath,".") for mp4 in mp4s]   #把mp4文件名替换成相对路径名


    for mp4 in mp4s:
        temp_wav_dir = os.path.basename(mp4).replace("mp4", "wav")
        output_file = output_wav_dir + temp_wav_dir
        ff = FFmpeg(
            inputs={mp4: None},
           # outputs={output_file: '-vn -ar 16000 -ac 2 -ab 192 -f wav'},
            outputs={output_file:None},

        )
        # subprocess.run(re.split(r"\s+", ff.cmd)) #我就奇怪了为啥这样运行不了

        ##改成把所有的命令存到sh文件中，再运行sh文件
        with open(sh_path,"a") as f:
            f.write(ff.cmd+";")









