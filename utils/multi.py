
from utils.bert import bert_data_collator
from utils.hubert import hubert_data_collator



def multi_data_collator(batch_list):
    """
    bert和hubert transformer里的datacollator
    :param batch_list:
    :return:
    """


    def popd(d):
        d.pop("audio_vec")
        return d
    hubert_batch_list = [{"input_values": d["audio_vec"]} for d in batch_list]  #注意这个inputvalues，要是不改名，这个datacollator工作不了
    bert_batch_list = [ popd(d) for d in batch_list ]

    audio_batch= hubert_data_collator(hubert_batch_list)
    text_batch = bert_data_collator(bert_batch_list)
    return text_batch,audio_batch
