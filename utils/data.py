from setting import setting
import pandas as pd
import pathlib

import librosa
from utils.hubert import feature_extractor
from torch.utils.data import Dataset
from utils.bert import tokenizer
import torch

##  代修改筛选

def get_data(data_use = "train",mode="multi"):
    data_path = setting.save_data_path / pathlib.Path("%s_%s.dt"%(mode,data_use))
    if not data_path.exists():
        print(f"  - Creating new {data_use} data")
        data = MELDDataSet(data_use=data_use,mode=mode)
        torch.save(data, data_path)
    else:
        print(f"  - Found cached {data_use} data")
        data = torch.load(data_path)

    return data




class MELDDataSet(Dataset):
    def __init__(self,data_use = "train",mode="audio"):
        """

        :param data_use:  train or dev or test
        :param mode:  sent or audio or multi     (单模态or多模态)
        """
        self.data_use = data_use
        self.mode = mode
        self.train_sent_data_path = setting.data_path / pathlib.Path("train_sent_emo.csv")
        self.dev_sent_data_path = setting.data_path / pathlib.Path("dev_sent_emo.csv")
        self.test_sent_data_path = setting.data_path / pathlib.Path("test_sent_emo.csv")

        self.train_audio_data_path = setting.data_path / pathlib.Path("MELD") / pathlib.Path("train_splits") / pathlib.Path("wav")
        self.dev_audio_data_path = setting.data_path / pathlib.Path("MELD") / pathlib.Path("dev_splits_complete") / pathlib.Path("wav")
        self.test_audio_data_path = setting.data_path / pathlib.Path("MELD") / pathlib.Path(" output_repeated_splits_test") / pathlib.Path("wav")

        self.df = self.load_data()
        self.wav_path_list = self.df["wav_path"].tolist()
        self.emotion_label = [setting.Emotion2ID[emotion] for emotion in self.df["Emotion"].tolist()]
        self.sentiment_label = [setting.Sentiment2ID[sentiment] for sentiment in self.df["Sentiment"].tolist()]

        if self.mode == "audio" or self.mode=="multi":
            self.audio = self.load_audio().tolist()
        if self.mode == "text" or self.mode=="multi":
            self.text_vec = self.load_text_vector().tolist()

        self.len = len(self.emotion_label)




    def load_data(self):

        path,wav_path = None,None
        if self.data_use =="train":
            path = self.train_sent_data_path
            wav_path = "train_splits/wav"  # 存训练集合的音频文件的目录
        elif self.data_use == "dev":
            path = self.dev_sent_data_path
            wav_path = "dev_splits_complete/wav"
        elif self.data_use == "test":
            path = self.test_sent_data_path
            wav_path = "output_repeated_splits_test/wav"

        df = pd.read_csv(path)
        df['wav_path'] = str(setting.data_path) + '/' + wav_path + '/dia' + df['Dialogue_ID'].map(str) + '_utt' + df[
            'Utterance_ID'].map(str) + '.wav'

        df = self.clean_data(df)
        df.drop('Dialogue_ID', axis=1, inplace=True)
        df.drop('Utterance_ID', axis=1, inplace=True)
        df.drop('StartTime', axis=1, inplace=True)
        df.drop('EndTime', axis=1, inplace=True)

        return df





    def clean_data(self,df):
        """
        数据清洗，
        有的utterence没有对应的wav文件
        我们就将这条数据给删除

        :return:
        """
        for index,row in df.iterrows():
            if not pathlib.Path(row["wav_path"]).exists():
                df = df.drop(index)
        return df# 我先筛选少一点，便于调试，到时候再改

    def load_audio(self):
        """
        将wav_path 转为向量
        :return:
        """

        #import swifter
        import time
        def map(wav_path):
            max_duration = 15
            speech, _ = librosa.load(wav_path, sr=16000, mono=True)
            input_values = feature_extractor(speech, sampling_rate=16000, padding=False,max_length=int(feature_extractor.sampling_rate * max_duration),truncation=True)
            return input_values
        time1 = time.time()
        #df2 = self.df["wav_path"].swifter.apply(map)
        df2 = self.df["wav_path"].apply(map)
        time2 = time.time()
        print(time2-time1)
        return df2
    def load_text_vector(self):
        """
        将text列表转为向量
        :return:
        """
        def map(sent):
            input_values  = tokenizer(sent,truncation=True)
            return input_values

        return self.df["Utterance"].apply(map)



    def __getitem__(self, index):
        if self.mode =="audio":

            # tokenize_inputs= feature_extractor(examples["speech"], sampling_rate=16000, padding=True
            #                                    ,max_length=int(feature_extractor.sampling_rate * max_duration),
            #                                    truncation=True)

            input_values = self.audio[index]
            input_values["emotion"] = self.emotion_label[index]
            input_values["sentiment"] = self.sentiment_label[index]
            if len(input_values["input_values"])!=1:  ##加这个判断的原因极为丑陋，看不懂直接跳过
                return input_values
            input_values["input_values"] = input_values["input_values"][0].tolist()
            return input_values
            #return input_values["input_values"],input_values["sentiment"]

        elif self.mode == "text":
            input_values = self.text_vec[index].data
            input_values["emotion"] = self.emotion_label[index]
            input_values["sentiment"] = self.sentiment_label[index]
            return input_values

        elif self.mode == "multi":
            input_values2 = self.audio[index]

            input_values = self.text_vec[index].data
            input_values["emotion"] = self.emotion_label[index]
            input_values["sentiment"] = self.sentiment_label[index]
            input_values["audio_vec"] = self.audio[index].data["input_values"][0].tolist()
            return input_values







    def __len__(self):
        return self.len







    # def load_data(self):
    #     train_dataset = Dataset.from_pandas(self.load_single_data(True,False,False))
    #     dev_dataset = Dataset.from_pandas(self.load_single_data(False,True,False))
    #     test_dataset = Dataset.from_pandas(self.load_single_data(False,False,True))
    #     dataset_dict = DatasetDict({"train":train_dataset,"dev":dev_dataset,"test":test_dataset})
    #     return dataset_dict








#################TEST####################################################

from transformers.feature_extraction_utils import BatchFeature
# def my_fn(batch):
#
#     batch = data_collator(batch)
#     return batch

if __name__ == "__main__":
    pass
    # base_path = str(setting.data_path)
    # datafiles = {'train':base_path+"/train_sent_emo.csv",'dev':base_path+"/dev_sent_emo.csv",'test':base_path+"/test_sent_emo.csv"}
    # dataset = load_dataset('csv',data_files=datafiles)
    # print(dataset)
    dataset = MELDDataSet(data_use="train",mode="multi")
    for i in dataset:
        print("dsa")
    # train_dataloader = DataLoader(
    #     dataset, shuffle=True, batch_size=8,collate_fn=my_fn
    # )
    # for batch in train_dataloader:
    #     print("ENDING")

