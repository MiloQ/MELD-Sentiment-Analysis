"""
此函数编写bert操作流程中需要的函数
"""

from setting import setting
from transformers import AutoConfig, AutoTokenizer,DataCollatorWithPadding
import numpy as np
from datasets import load_metric
import pathlib
import itertools
from sklearn.metrics import classification_report
from utils.common import compute_weighted_f1


config = AutoConfig.from_pretrained(setting.config_path)
tokenizer = AutoTokenizer.from_pretrained(setting.tokenizer_path, config=config,add_prefix_space=True) #add_prefix_space参数，给开头单词添加空格，很重要
bert_data_collator= DataCollatorWithPadding(tokenizer=tokenizer)



def tokenize_funtion(examples):
    tokenize_input = tokenizer(examples["Utterance"],truncation=True)
    Emotion_labels = list()
    Sentiment_labels = list()

    for emotion in examples["Emotion"]:
        Emotion_labels.append(setting.Emotion2ID[emotion])
    for sentiment in examples["Sentiment"]:
        Sentiment_labels.append(setting.Sentiment2ID[sentiment])
    tokenize_input["Emotion_labels"] = Emotion_labels
    tokenize_input["Sentiment_labels"] = Sentiment_labels
    return tokenize_input




def create_tokenized_dataset(dataset):
    raw_dataset = dataset.df_dict
    tokenized_datasets = raw_dataset.map(tokenize_funtion,batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("Sentiment_labels", "labels")#把Sentiment作为先训练的目标来测试一下
    return tokenized_datasets

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)


    true_labels = [setting.Sentiment_list[label] for label in labels ]
    true_predictions = [setting.Sentiment_list[prediction] for prediction in predictions]
    metric_dict = classification_report(y_true=true_labels,
                                        y_pred=true_predictions,
                                        target_names=setting.Sentiment_list,
                                        output_dict=True)
    weighted_f1 = compute_weighted_f1(metric_dict, target_names=setting.Sentiment_list)
    result_dict = {"weighted_f1": weighted_f1}
    for tag in setting.Sentiment_list:
        for metric_name, metric_val in metric_dict[tag].items():
            if metric_name == "support": continue
            result_dict[f"{tag}_{metric_name}"] = metric_val

    return result_dict



##################prediction & eval####################################


def predict_model(trainer, tokenized_datasets, name="dev"):

    # makei predictions
    # predictions: (dataset_size, max_len, num_classes)
    # label_ids: (dataset_size, max_len)
    predictions, labels, metrics = trainer.predict(tokenized_datasets[name])
    predictions = np.argmax(predictions, axis=-1)

    y_true = [setting.Sentiment_list[label] for label in labels]
    y_pred = [setting.Sentiment_list[prediction] for prediction in predictions]

    return y_true, y_pred


def evaluate_model(trainer, tokenized_datasets, name="dev"):
    label_list = setting.Sentiment_list

    y_true, y_pred = predict_model(trainer, tokenized_datasets, name=name)

    y_pred_list = y_pred
    y_true_list = y_true

    metric = classification_report(y_true=y_true_list, y_pred=y_pred_list, target_names=label_list, output_dict=True)
    metric_str = classification_report(y_true=y_true_list, y_pred=y_pred_list, target_names=label_list, output_dict=False)

    return metric, metric_str