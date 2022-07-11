import torch
from transformers import AdamW,get_scheduler
from setting import setting
from utils.data import MELDDataSet,get_data
from utils.common import  prepare_logger
from torch.utils.data import DataLoader
from utils.multi import multi_data_collator

from Model.MultiModal1 import MultiModal1
from Model.MultiModal2 import MultiModal2
from Model.MultiModal3 import MultiModal3
from sklearn.metrics import classification_report
from accelerate import Accelerator
import numpy as np
import pathlib
import random
import os

"""
过时文件，尝试使用混合精度训练
"""

from torch.cuda.amp import autocast as autocast,GradScaler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


#多卡

# dist.init_process_group(backend='nccl')
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)


## 一些基础设置
SEED =  43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

##  超参数设置
lr = 1e-5
n_epoch = 20
batch_size = 8
accumulation_steps = 8 #  梯度累积  模拟batchsize=8*8





### logger模块
settings = [ "classfication=%s" % setting.mode,
             "lr=%f" %lr,
             "n_epoch=%d" % n_epoch,
             'batch_size=%d'%batch_size,
             "seed=%d"%SEED,
             "mode=test",
             "info=dim=4"

            ]
log_path = pathlib.Path("logs/")
if not log_path.exists(): log_path.mkdir()
logger = prepare_logger(filename="%s/logfile@%s.log" % (log_path, "_".join(settings)), name="audio2")
logger.info("\n".join(settings))





## 获取Dataloader
train_dataset = get_data(data_use="train",mode="multi")
dev_dataset = get_data(data_use="dev",mode="multi")
test_dataset = get_data(data_use="test",mode="multi")
#train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(
    train_dataset,batch_size=batch_size,collate_fn=multi_data_collator
)
eval_dataloader = DataLoader(
    dev_dataset,batch_size =batch_size,collate_fn=multi_data_collator
)
test_dataloader = DataLoader(
    test_dataset,batch_size=batch_size,collate_fn=multi_data_collator
)

#tokonized_dataset = tokenized_dataset_for_hubert(dataset)






## 加载模型

model = MultiModal3()
optimizer = AdamW(model.parameters(), lr=lr)
num_training_steps = n_epoch * len(train_dataloader)
lr_scheduler = get_scheduler(          ## 控制学习率的scheduler
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)




### 多卡训练
# accelerator = Accelerator()
# train_dataloader, eval_dataloader,test_dataloadert ,model, optimizer = accelerator.prepare(
#      train_dataloader, eval_dataloader, test_dataloader,model, optimizer
#  )

## 混合精度训练
scaler = GradScaler()


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
logger.info(device)

## Training   And   Val  Loops
print("Training")

for epoch in range(n_epoch):
    model.train()
    for text_batch,audio_batch in train_dataloader:

        text_batch.to(device)
        audio_batch.to(device)
        with autocast():
            outputs = model(text_batch,audio_batch)
            Cross_loss = model.cal_loss(outputs,text_batch["emotion"])
        scaler.scale(Cross_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # optimizer the net

        lr_scheduler.step()
        optimizer.zero_grad()  # rese


        ##收集内存
        # gc.collect()
        # torch.cuda.empty_cache()


    ###Val
    model.eval()
    labels = []
    preds = []
    for text_batch,audio_batch in test_dataloader:
        text_batch.to(device)
        audio_batch.to(device)
        with torch.no_grad():
            outputs = model(text_batch,audio_batch)

        predictions = torch.argmax(outputs, dim=-1)
        preds.append(predictions.cpu().numpy().tolist())
        labels.append(text_batch["emotion"].cpu().numpy().tolist())

    labels = sum(labels,[])  # 转化为一维列表
    preds = sum(preds,[])
    report = classification_report(labels, preds,target_names=setting.Emotion_list,digits=4)
   #report = classification_report(labels, preds)
    logger.info(report)
    print(epoch)



## Save_checkpoint

# torch.save(model.state_dict(),setting.text_checkpoint_path)


## TEST





























