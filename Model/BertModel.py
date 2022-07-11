import torch.nn as nn
import torch
from transformers import AutoConfig,BertModel
from setting import setting
import torch.nn.functional as F



class BertForSequenceClassifaction(nn.Module):
    def __init__(self,drop=0.2,label_num=3):
        super(BertForSequenceClassifaction,self).__init__()
        self.config = AutoConfig.from_pretrained(setting.config_path)
        self.pretrained_model = BertModel.from_pretrained(setting.model_path)
        self.fc_dropout = nn.Dropout(drop)
        self.fc = nn.Linear(768,label_num)  # BERT最后一层输出是768
        self.criterion = nn.CrossEntropyLoss()


    def forward(self,inputs): #直接把batch的字典给送进去
        #  我们不送进去labels，只送input_ids 和attention_mask

        outputs = self.pretrained_model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["attention_mask"],
                                        token_type_ids=inputs["token_type_ids"],
                                        return_dict=True)

        preds = self.fc(self.fc_dropout(torch.mean(outputs.get('last_hidden_state'), 1)))
        return F.log_softmax(preds,dim=1)


    def cal_loss(self,preds,labels):
        #将labels 变成one-hot 编码
        one_hot_labels = torch.eye(len(setting.Emotion_list))[labels,:].cuda()
        return self.criterion(preds,one_hot_labels)

# if __name__ == "__main__":
#     model = HubertForSequenceClassfication()
#     print(model.config.return_attention_mask)
#







