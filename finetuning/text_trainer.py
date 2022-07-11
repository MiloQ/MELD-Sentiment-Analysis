from transformers import Trainer
from transformers import AutoModelForSequenceClassification,TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from setting import setting
from utils.data import MELDDataSet
from utils.bert import create_tokenized_dataset,config,tokenizer,data_collator,compute_metrics,evaluate_model
import torch
import random
import numpy as np
import pathlib
import wandb
from utils.common import prepare_logger,remove_directories,compute_weighted_f1
import re
import os
import pickle
import shutil

"""
本文件为过时文件，在研究text模态时用的huggingface trainer所做
"""




### 一些设置
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
use_checkpoint = False
use_hpo_search = False #是否进行超参数搜索





model = AutoModelForSequenceClassification.from_pretrained(setting.model_path)
dataset = MELDDataSet()
tokenized_datasets = create_tokenized_dataset(dataset)


##log part

settings = [ "classfication=%s" % setting.mode,
            "model=%s" % setting.model_name,
            ]
log_path = pathlib.Path("logs/")
if not log_path.exists(): log_path.mkdir()
logger = prepare_logger(filename="%s/logfile@%s.log" % (log_path, "_".join(settings)), name="t")
logger.info("\n".join(settings))



def hp_space_optuna(trial):  #超参数搜索空间
    return {
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-6, 5e-6, 1e-5]),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [3, 5]),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [2]),
    }
def hpo(tokonized_datasets):  # 超参数搜索函数，搜到最好的超参数
    hpo_path = pathlib.Path("hpo")
    if not hpo_path.exists(): hpo_path.mkdir()
    hpo_path = hpo_path / pathlib.Path("hpo.pkl")

    if not pathlib.Path(hpo_path).exists():
        logger.info("HYPERPARAMETER SEARCH")
        model_init = lambda: AutoModelForSequenceClassification.from_pretrained(setting.model_path)
        trainer = Trainer(args=TrainingArguments(output_dir="output/hpo", evaluation_strategy="epoch", eval_steps=500,
                                                 report_to="none", disable_tqdm=True),
                          tokenizer=tokenizer,
                          train_dataset=tokenized_datasets["train"],
                          eval_dataset=tokenized_datasets["dev"],
                          data_collator=data_collator,
                          model_init=model_init,
                          compute_metrics=compute_metrics,)

        best_trail = trainer.hyperparameter_search(hp_space=hp_space_optuna,
                                                   # A function that defines the hyperparameter search space. Will default to default_hp_space_optuna() or default_hp_space_ray() or default_hp_space_sigopt() depending on your backend.
                                                   direction="maximize",
                                                   backend="optuna",
                                                   n_trials=6)

        logger.info("CLEANUP")
        remove_directories(["runs/", "output/"])

        hp_dict = dict()

        hp_dict["lr"] = best_trail.hyperparameters["learning_rate"]
        hp_dict["batch_size"] = best_trail.hyperparameters["per_device_train_batch_size"]
        hp_dict["n_epoch"] = best_trail.hyperparameters["num_train_epochs"]


        with open(hpo_path, "wb") as fp:
            pickle.dump(hp_dict, fp)
    else:
        logger.info("READING ALREADY SEARCHED HYPERPARAMETERS")
        with open(hpo_path, "rb") as fp:
            hp_dict = pickle.load(fp)

    return hp_dict
def train_model(trainer):
    # load checkpoint and train
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and use_checkpoint:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logger.info(f"checkpoint detected, resuming training at {last_checkpoint}")

    # if use_checkpoint == False, then checkpoint == None, no checkpoints will be loaded
    if last_checkpoint is not None:
        checkpoint = last_checkpoint
    else:
        checkpoint = None

    trainer.train(resume_from_checkpoint=checkpoint)

def get_checkpoint(folder,tokonized_datasets,mode="best"):
    """
    :param folder: 从存储checkpoint的文件夹里找
    :param tokonized_datasets:
    :param mode:
    :return:
    """
    assert mode in ["best", "median", "mean", "worst"] #只支持这几种模式
    checkpoint_name_pattern = re.compile(r"^" + "checkpoint" + r"\-(\d+)$")
    checkpoints = [os.path.join(folder, path) for path in os.listdir(folder) if (checkpoint_name_pattern.search(path) is not None) and
                                                                                os.path.isdir(os.path.join(folder, path))]

    checkpoint_dict = dict()

    for checkpoint in checkpoints:
        logger.info("evaluating checkpoint: %s..." % checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        trainer = Trainer(model=model,
                          args=training_args,
                          tokenizer=tokenizer,
                          train_dataset=tokenized_datasets["train"],
                          eval_dataset=tokenized_datasets["dev"],
                          data_collator=data_collator,
                          compute_metrics=compute_metrics)
        metric, metric_str = evaluate_model(trainer, tokenized_datasets, name="dev")
        weighted_f1 = compute_weighted_f1(metric, target_names=setting.Sentiment_list)
        checkpoint_dict[checkpoint] = weighted_f1
        perf_str = "\n".join("\t%s: %.4f" % (key, val)
                             for key, val in sorted(checkpoint_dict.items(), key=lambda x: x[1], reverse=True))
        logger.info(perf_str)


    # select checkpoints based on different criterion
    if mode == "best":
        checkpoint = max(checkpoint_dict, key=checkpoint_dict.get)
    else:
        weighted_f1_arr = np.fromiter(checkpoint_dict.values(), dtype=float, count=len(checkpoint_dict))
        if mode == "mean":
            mean = np.mean(weighted_f1_arr)
            checkpoint_dict = {key: abs(val - mean) for key, val in checkpoint_dict.items()}
        if mode == "median":
            median = np.median(weighted_f1_arr)
            checkpoint_dict = {key: abs(val - median) for key, val in checkpoint_dict.items()}
        # if mode == "worst", no need to modify the checkpoint_dict
        checkpoint = min(checkpoint_dict, key=checkpoint_dict.get)

    return checkpoint




def save_best_checkpoint(checkpoint, save=True):
    # checkpoint: the filename of best checkpoint during evaluation
    if not save:
        logger.info("NOT SAVING TRAINING CHECKPOINT")
        return

    logger.info("SAVING THE BEST CHECKPOINT DURING TRAINING")
    save_checkpoint_path = setting.checkpoint_path
    if not save_checkpoint_path.exists(): save_checkpoint_path.mkdir()


    for filename in os.listdir(checkpoint):
        shutil.move(os.path.join(checkpoint, filename), save_checkpoint_path)




# search for best main checkpoint and hyperparameters
if use_hpo_search:
    hp_dict = hpo(tokenized_datasets)  #返回一个最好的记录超参数的字典
else:
    hp_dict = None
lr = hp_dict["lr"] if hp_dict is not None else 1e-5
batch_size = hp_dict["batch_size"] if hp_dict is not None else 32
n_epoch = hp_dict["n_epoch"] if hp_dict is not None else 3

##wandb

param = {"lr": lr,
         "n_epoch": n_epoch,
         'batch_size':batch_size,
         "seed": SEED,
         }
logger.info("\n".join(param))
wandb.init(project="sentiment-analysis",
           name="run",
           config=param)

training_args = TrainingArguments(
    "bert-sentiment-text-finetune",
    learning_rate=lr,
    num_train_epochs=n_epoch,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    fp16 = True,
    do_train=True,
    do_eval=True,
    #checkpoint setting
    save_strategy="steps",
    save_steps=100,
    # evaluation
    evaluation_strategy="steps",
    eval_steps=100,
    disable_tqdm=True, #不要进度条了
    report_to="wandb",
)



trainer = Trainer(
    model,
    training_args,

    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)





### Train##########################################
logger.info("Start Training")
train_model(trainer)

### validation ##################################
logger.info("VALIDATION: validate checkponints")
best_checkpoint = get_checkpoint(training_args.output_dir,
                                 tokenized_datasets,
                                 mode="best",
                                 )
logger.info("BEST CHECKPOINT: %s" % best_checkpoint)



###TEST################################################
model = AutoModelForSequenceClassification.from_pretrained(best_checkpoint)
trainer = Trainer(
    model,
    training_args,
    # logging_strategy="epoch",
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


metric, metric_str = evaluate_model(trainer, tokenized_datasets, name="test")
weighted_f1 = compute_weighted_f1(metric, target_names=setting.Sentiment_list)
logger.info(metric_str)
logger.info("METRIC: weighted F1=%.4f" % weighted_f1)

save_best_checkpoint(best_checkpoint, save=True)








