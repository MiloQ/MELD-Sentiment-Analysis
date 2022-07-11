from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from setting import setting
from transformers import HubertForSequenceClassification,Wav2Vec2FeatureExtractor,HubertModel

"""
Download  and save the pretrained model 
"""


## text Part
config = AutoConfig.from_pretrained(setting.model_name, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(setting.model_name, config=config)#因为想做一句/一段话的情感分类，所以是sequence
tokenizer = AutoTokenizer.from_pretrained(setting.model_name)

config.save_pretrained(setting.config_path)
model.save_pretrained(setting.model_path)
tokenizer.save_pretrained(setting.tokenizer_path)



#AudioPart

AudioConfig= AutoConfig.from_pretrained(setting.AudioModel_name)
AudioModel = HubertModel.from_pretrained(setting.AudioModel_name)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(setting.AudioModel_name)

AudioConfig.save_pretrained(setting.AudioConfig_path)
AudioModel.save_pretrained(setting.AudioModel_path)
feature_extractor.save_pretrained(setting.feature_extractor_path)
