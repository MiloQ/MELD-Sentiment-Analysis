from Model.BertModel import BertForSequenceClassifaction
from Model.HubertModel import HubertForSequenceClassfication
from Model.MultiModal1 import MultiModal1
from Model.MultiModal2 import MultiModal2
from Model.MultiModal3 import MultiModal3
from transformers import AdamW,get_scheduler
from setting import setting
from sklearn.metrics import classification_report

import torch


def train_model(cfg,train_dataloader,eval_dataloader,test_dataloader):


    ## model Part

    if cfg.mode=="text":  # Text Modal
        model = BertForSequenceClassifaction(label_num=7)
    elif cfg.mode == "audio":
        model = HubertForSequenceClassfication(label_num=7)
    elif cfg.mode == "multi":
        multi_models = {'MultiModal1': MultiModal1,
                        'MultiModal2': MultiModal2,
                        'MultiModal3': MultiModal3}
        model = multi_models[cfg.multi_model]()



    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    num_training_steps = cfg.n_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(  ## 控制学习率的scheduler
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print("Training")

    for epoch in range(cfg.n_epoch):
        model.train()
        for batch in train_dataloader: # batch可能是一个textbatch，audiobatch，或者在multi模式下的（textbatch，audiobatch）
            if cfg.mode == "text" or cfg.mode == "audio":
                batch.to(device)
                outputs = model(batch)
            elif cfg.mode == "multi":
                text_batch = batch[0]
                audio_batch = batch[1]
                text_batch.to(device)
                audio_batch.to(device)
                outputs = model(text_batch, audio_batch)
                batch = text_batch  # 为了下一行的batch["emotion"]表现和text，audio一样

            Cross_loss = model.cal_loss(outputs, batch["emotion"])
            Cross_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()



        model.eval()
        labels = []
        preds = []
        for batch in test_dataloader:
            if cfg.mode == "text" or cfg.mode == "audio":
                batch.to(device)
                with torch.no_grad():
                    outputs = model(batch)
            elif cfg.mode == "multi":
                text_batch,audio_batch = batch[0],batch[1]
                text_batch.to(device)
                audio_batch.to(device)
                with torch.no_grad():
                    outputs = model(text_batch, audio_batch)


            predictions = torch.argmax(outputs, dim=-1)
            preds.append(predictions.cpu().numpy().tolist())
            labels.append(batch["emotion"].cpu().numpy().tolist())

        labels = sum(labels, [])  # 转化为一维列表
        preds = sum(preds, [])
        report = classification_report(labels, preds, target_names=setting.Emotion_list,digits=4)
        print(report)
        print(epoch)