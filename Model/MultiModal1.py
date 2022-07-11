import torch.nn as nn
import torch
from transformers import AutoConfig,HubertModel,BertModel
from setting import setting
import torch.nn.functional as F

"""
多模态1 使用
bert和hubert提取特征之后 
将text和audio特征concat之后
送入分类网络
"""

class MultiModal1(nn.Module):
    def __init__(self,drop=0.25,label_num=7):
        super(MultiModal1,self).__init__()
        self.text_pretrained_model = BertModel.from_pretrained(setting.model_path)
        self.audio_pretrained_model = HubertModel.from_pretrained(setting.AudioModel_path)
        self.fc_dropout = nn.Dropout(drop)
        self.fc1 = nn.Linear(2*768,1200)
        self.fc2 = nn.Linear(1200,600)
        self.fc3 = nn.Linear(600,300)
        self.fc4 = nn.Linear(300,label_num)
        self.fc = nn.Linear(2*768,label_num)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_batch,audio_batch):  # 直接把batch的字典给送进去

        ## Bert部分网络
        text_outputs= self.text_pretrained_model(input_ids=text_batch["input_ids"],
                                                 attention_mask = text_batch["attention_mask"],
                                                 token_type_ids = text_batch["token_type_ids"],
                                                 return_dict = True
                                                 )
        text_feature = torch.mean(text_outputs.get('last_hidden_state'),1)
        ## Hubert 部分网络
        audio_outputs = self.audio_pretrained_model(
            input_values=audio_batch["input_values"],
            return_dict=True
        )
        audio_feature = torch.mean(audio_outputs.get('last_hidden_state'),1)
        concat_feature = torch.concat((text_feature,audio_feature),dim=1)

        l1 = self.fc1(self.fc_dropout(concat_feature))
        l2 = self.fc2(self.fc_dropout(l1))
        l3 = self.fc3(l2)
        preds = self.fc4(l3)

        # preds = self.fc3(self.fc_dropout(l2))
        #preds = self.fc(self.fc_dropout(concat_feature))
        return F.log_softmax(preds, dim=1)

    def cal_loss(self, preds, labels):
        # 将labels 变成one-hot编码
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one_hot_labels = torch.eye(len(setting.Emotion_list))[labels, :]
        preds_cpu = preds.cpu()
        return self.criterion(preds_cpu, one_hot_labels)







if __name__ == "__main__":
    model = MultiModal1()





