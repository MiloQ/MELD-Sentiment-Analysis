import torch.nn as nn
import torch
from transformers import AutoConfig,HubertModel
from setting import setting
import torch.nn.functional as F



class HubertForSequenceClassfication(nn.Module):
    def __init__(self,drop=0.2,fc_hiddensize=768,label_num=7):
        super(HubertForSequenceClassfication,self).__init__()


        self.config = AutoConfig.from_pretrained(setting.AudioConfig_path)
        self.pretrained_model = HubertModel.from_pretrained(setting.AudioModel_path)
        self.fc_dropout = nn.Dropout(drop)
        self.fc = nn.Linear(fc_hiddensize,label_num)
        self.criterion = nn.CrossEntropyLoss()


    def forward(self,inputs): #直接把batch的字典给送进去



        outputs = self.pretrained_model(input_values=inputs["input_values"],
                                         return_dict=True)
        preds = self.fc(self.fc_dropout(torch.mean(outputs.get('last_hidden_state'), 1)))
        return F.log_softmax(preds,dim=1)


    def cal_loss(self,preds,labels):
        #将labels 变成one-hot编码
        #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one_hot_labels = torch.eye(len(setting.Emotion_list))[labels,:].cuda()

        return self.criterion(preds,one_hot_labels)

# if __name__ == "__main__":
#     model = HubertForSequenceClassfication()
#     print(model.config.return_attention_mask)
#







