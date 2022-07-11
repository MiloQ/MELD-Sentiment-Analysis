from setting import setting
from transformers import Wav2Vec2FeatureExtractor,DataCollatorWithPadding
import librosa
from transformers.feature_extraction_utils import BatchFeature


"""
在语音领域，我们的思路是先用wav2Vec2 将wav文件的序列表示转化为Vector 类似NLP中做初始的Embedding
然后送入Hubert中Finetune获得更好的向量表示
然后将向量送入下游任务中

"""



## 给训练的东西准备一些加载器
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(setting.feature_extractor_path)
hubert_data_collator = DataCollatorWithPadding(tokenizer=feature_extractor)









def map_to_array(example):
    speech, _ = librosa.load(example["wav_path"], sr=16000, mono=True)
    # speech  = librosa.resample(speech.astype(np.float32), 16000)
    example["speech"] = speech
    return example

def tokenized_function(examples):
    max_duration = 3
    # tokenize_inputs= feature_extractor(examples["speech"], sampling_rate=16000, padding=True
    #                                    ,max_length=int(feature_extractor.sampling_rate * max_duration),
    #                                    truncation=True)
    #tokenize_inputs = feature_extractor(examples["speech"],sampling_rate=16000,padding=False)
    #tokenize_inputs2 = feature_extractor(None)
    #tokenize_inputs3 = BatchFeature()
    Emotion_labels = [setting.Emotion2ID[i] for i in examples["Emotion"]]
    Sentiment_labels = [setting.Sentiment2ID[i] for i in examples["Sentiment"]]

    examples["Emotion_labels"] = Emotion_labels
    examples["Sentiment_labels"] = Sentiment_labels
    examples = BatchFeature(examples)
    return examples

def tokenized_dataset_for_hubert(dataset):
    raw_dataset = dataset.df_dict
   # raw_dataset = raw_dataset.map(map_to_array)
    tokenized_datasets = raw_dataset.map(tokenized_function,batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["Emotion",  "Episode", "Season", "Sentiment", "Speaker", "Sr No.", "Utterance",
         ])
    tokenized_datasets.remove_columns(["wav_path"])
    tokenized_datasets["train"].remove_columns(['__index_level_0__'])
    tokenized_datasets["dev"].remove_columns(['__index_level_0__'])
    tokenized_datasets = tokenized_datasets.rename_column("Emotion_labels","labels")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


