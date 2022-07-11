from setting import config
from utils.data import get_data
from torch.utils.data import DataLoader
from utils.multi import multi_data_collator
from finetuning.train import train_model
from utils.bert import bert_data_collator
from utils.hubert import hubert_data_collator
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cfg = config.DefaultConfig()




print("Start Loading Data........")
train_dataset = get_data(data_use="train",mode=cfg.mode)
dev_dataset = get_data(data_use="dev",mode=cfg.mode)
test_dataset = get_data(data_use="test",mode=cfg.mode)

if cfg.mode == "multi":
    train_dataloader = DataLoader(
        train_dataset,batch_size=cfg.batchsize,collate_fn=multi_data_collator
    )
    eval_dataloader = DataLoader(
        dev_dataset,batch_size = cfg.batchsize,collate_fn=multi_data_collator
    )
    test_dataloader = DataLoader(
        test_dataset,batch_size= cfg.batchsize,collate_fn=multi_data_collator
    )
elif cfg.mode == "text":
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batchsize,collate_fn=bert_data_collator
    )
    eval_dataloader = DataLoader(
        dev_dataset, batch_size=cfg.batchsize, collate_fn=bert_data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batchsize, collate_fn=bert_data_collator
    )

elif cfg.mode == "audio":
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batchsize,collate_fn=hubert_data_collator
    )
    eval_dataloader = DataLoader(
        dev_dataset, batch_size=cfg.batchsize,collate_fn=hubert_data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg.batchsize,collate_fn=hubert_data_collator
    )



if __name__ == "__main__":
    train_model(cfg,train_dataloader,eval_dataloader,test_dataloader)