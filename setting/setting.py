import pathlib

#base_path = pathlib.Path("/data/qzy/meld-sentiment-analysis") #得按服务器件上的目录结构来写代码
base_path = pathlib.Path("/home/user/qzy/meld-sentiment-analysis")
pretrained_path = base_path / pathlib.Path("pretrained")
data_path = base_path / pathlib.Path("data") / pathlib.Path("MELD")



## 这个文件夹放置处理好的data，比如把音频转化为数组，text转化为tokonid
save_data_path = base_path / pathlib.Path("data") / pathlib.Path("saved_data")




# Bert
model_name = "bert-base-uncased"


## Audio Part
#AudioModel_name = "superb/hubert-large-superb-er"
AudioModel_name = "ntu-spml/distilhubert"
#AudioModel_name = "facebook/wav2vec2-base"


## task Mode
## 我们的实验顺序是从sentiment开始研究，到Emotion，最终想做结合sentiment的标签信息去研究Emotion

mode = "sentiment"   ##我们的任务分类 总共有两种mode，分类Emotion or 分类Sentiment

##############################################################################
##labsels

Emotion_list = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
Emotion2ID = {"anger":0,"disgust":1,"fear":2,"joy":3,"neutral":4,"sadness":5,"surprise":6}
Sentiment_list = ["neutral","positive","negative"]
Sentiment2ID = {"neutral":0,"positive":1,"negative":2}


if mode=="sentiment":
    labels = Sentiment_list
    label2id = Sentiment2ID
elif mode=="emotion":
    labels = Emotion_list
    label2id = Emotion2ID





####################################################################################
# model path
tokenizer_path = str(pretrained_path / pathlib.Path(f"{model_name}/tokenizer"))
config_path = str(pretrained_path / pathlib.Path(f"{model_name}/config"))
model_path = str(pretrained_path / pathlib.Path(f"{model_name}/model"))
text_checkpoint_path = str(pretrained_path / pathlib.Path(f"{model_name}/checkpoint"))    #最好的checkpoint

## Audio Model Path
AudioModel_path = str(pretrained_path / pathlib.Path(f"{AudioModel_name}/model"))
feature_extractor_path = str(pretrained_path / pathlib.Path(f"{AudioModel_name}/feature_extractor"))
AudioConfig_path = pretrained_path / pathlib.Path(f"{AudioModel_name}/config")


