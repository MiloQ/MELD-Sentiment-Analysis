"""
本个multimodal 使用crossmodal attention技术
"""

import torch.nn as nn
from transformers import BertModel,HubertModel
from setting import setting
import torch
import torch.nn.functional as F
from Model.transformer import TransformerEncoder


class MultiModal3(nn.Module):
    def __init__(self,labels = 7):
        super(MultiModal3,self).__init__()
        self.training = True
        self.attn_dropout = 0.3
        self.attn_dropout_a  = 0.1
        self.relu_dropout = 0.1
        self.res_dropout = 0.1
        self.embed_dropout = 0.1
        self.out_dropout = 0.1
        self.num_heads = 5
        self.attn_mask = False
        self.layers = 5
        self.outputdim = labels
        self.text_pretrained_model = BertModel.from_pretrained(setting.model_path)
        self.audio_pretrained_model = HubertModel.from_pretrained(setting.AudioModel_path)
        self.origin_d_text,self.origin_d_audio = 768,768 #经过两个预训练网络送出来的特征是768维度的
        self.d_text,self.d_audio = 30,30 # 送入之后的去做cross modal attention的我们用少一点的维度去训练
        self.combined_dim= self.d_text+self.d_audio
        # 1. Temporal convolutional layers
        self.proj_text = nn.Conv1d(self.origin_d_text, self.d_text, kernel_size=1, padding=0, bias=False)
        self.proj_audio= nn.Conv1d(self.origin_d_audio, self.d_audio, kernel_size=1, padding=0, bias=False)


        # 2. Cross Modal Attention

        self.trans_text_with_audio = self.get_network(self_type='ta')
        self.trans_audio_with_text = self.get_network(self_type='at')

        # 3. Self Attention
        self.trans_t_mem = self.get_network(self_type='t_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)

        self.fc = nn.Linear(self.combined_dim,self.combined_dim)
        self.fc0 = nn.Linear(self.combined_dim,self.combined_dim)
        self.fc1 = nn.Linear(self.combined_dim,200)
        self.fc2 = nn.Linear(200,self.outputdim)



        self.criterion = nn.CrossEntropyLoss()



    def get_network(self, self_type='t', layers=-1):
        if self_type in ['t', 'at', 'vt']:
            embed_dim, attn_dropout = self.d_text, self.attn_dropout
        elif self_type in ['a', 'ta', 'va']:
            embed_dim, attn_dropout = self.d_audio, self.attn_dropout_a

        elif self_type == 't_mem':
            embed_dim, attn_dropout = self.d_text, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_audio, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, text_batch, audio_batch):  # 直接把batch的字典给送进去

        ## Bert部分网络
        text_outputs = self.text_pretrained_model(input_ids=text_batch["input_ids"],
                                                  attention_mask=text_batch["attention_mask"],
                                                  token_type_ids=text_batch["token_type_ids"],
                                                  return_dict=True
                                                  )
        text_feature = text_outputs.get('last_hidden_state',1)
        ## Hubert 部分网络
        audio_outputs = self.audio_pretrained_model(
            input_values=audio_batch["input_values"],
            return_dict=True
        )
        audio_feature = audio_outputs.get('last_hidden_state',1)
        ##  此时text_feature 和audio_feature为 [batch_size,seqLen,embed_dim]


        text_feature = F.dropout(text_feature.transpose(1,2),p=self.embed_dropout,training=self.training)
        audio_feature = audio_feature.transpose(1,2)

        # 将feature 变成 [batch_size,embed_dim,seqLen] 然后送入卷积层
        proj_text_feature = self.proj_text(text_feature)
        proj_audio_feature = self.proj_audio(audio_feature)

        proj_text_feature = proj_text_feature.permute(2,0,1)
        proj_audio_feature = proj_audio_feature.permute(2,0,1)
        # now feature size 为[seqLen,batchsize,embedding]  然后送入attention模块

        text_with_audio = self.trans_text_with_audio(proj_text_feature,proj_audio_feature,proj_audio_feature)
        ta_selattn = self.trans_t_mem(text_with_audio)
        last_ta = ta_selattn[-1]  ##取最后的output去预测
        audio_with_text = self.trans_audio_with_text(proj_audio_feature,proj_text_feature,proj_text_feature)
        at_selattn = self.trans_a_mem(audio_with_text)
        last_at = at_selattn[-1]

        last_hidden_state = torch.concat([last_ta,last_at],dim=1)

        # A residual block
        last_hs_proj = self.fc0(F.dropout(F.relu(self.fc(last_hidden_state)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hidden_state

        output1 = F.dropout(self.fc1(last_hs_proj),p=0.2)
        preds = self.fc2(output1)
        return F.log_softmax(preds, dim=1)















    def cal_loss(self, preds, labels):
        # 将labels 变成one-hot编码
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        one_hot_labels = torch.eye(len(setting.Emotion_list))[labels, :]

        return self.criterion(preds, one_hot_labels.cuda())
