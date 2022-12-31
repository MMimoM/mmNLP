#%% libraries
from mmNLP.ClassificationModule import TxTClassification

import torch
import pandas as pd
import re


#%% load data
df = pd.read_excel('C:/Users/Mehya/OneDrive/enowa/NLP/Textdaten/preproc.xlsx')
txt = df.original_text.fillna(df.text[df.original_text.isnull()])
label = df.Label2

def preproc(txt):
    txt = txt.replace('Ã¤', 'ae')
    txt = txt.replace('Ãœ', 'ue')
    txt = txt.replace('Ã¼', 'ue')
    txt = txt.replace('Ã¶', 'oe')
    txt = txt.replace('ÃŸ', 'ss')
    txt = txt.replace('â', 'ae')
    txt = txt.replace('â€¢', '')
    txt = txt.lower()
    txt = txt.replace('Â§', 'paragraph')
    txt = txt.replace('%', 'prozent')
    txt = re.sub('\W+', ' ', txt)
    txt = re.sub(r'[0-9]', ' ', txt)
    txt = " ".join(txt.split())
    return txt

#%% Ckassification
embedding_parameter = {'model_type':'fasttext',
                       'embedding_size':50, 
                       'window_size':5, 
                       'subword_ngrams_min':3, 
                       'subword_ngrams_max':5}

nn_parameter = {'embedding_size':50,
                'architecture':'Kim2014',
                'batch_size':50,
                'epochs':10,
                'dropout':0.5,
                'learning_rate':0.01,
                'train_embedding':True,
                'num_filters':[5,5,5],
                'filter_sizes':[2,3,4]}

TxTClassificationObject = TxTClassification(txt_lines = txt,
                                            label = label,
                                            path = 'C:/Users/Mehya/OneDrive/enowa/NLP/src/enlp',
                                            use_pretrained_embedding = True,
                                            load_pytorch_classifier=False,
                                            padding=True,
                                            maxlen = 150,
                                            loss = torch.nn.CrossEntropyLoss(),
                                            optimizer = 'adam',
                                            embedding_parameter = embedding_parameter,
                                            nn_parameter = nn_parameter, 
                                            preprocessor = preproc)

TxTClassificationObject.nnSentenceClassification(metric = 'f1_score', save_pytorch_classifier=True)
report = TxTClassificationObject.evaluate(report_classification = True, out_of_sample=True)

#%% individuell nn architecture
embedding_parameter = {'model_type':'fasttext',
                       'embedding_size':400, 
                       'window_size':7, 
                       'subword_ngrams_min':3, 
                       'subword_ngrams_max':6}

nn_parameter = {'embedding_size':400,
                'architecture':'SimpleCNN',
                'batch_size':30,
                'epochs':10,
                'dropout':0.5,
                'learning_rate':0.001,
                'train_embedding':True,
                'num_filters':300,
                'filter_sizes':4}

TxTClassificationObject_2 = TxTClassification(TxTobject=TxTClassificationObject.TxTobject,
                                             eTxTobject_train = TxTClassificationObject.eTxTobject_train,
                                             eTxTobject_test = TxTClassificationObject.eTxTobject_test,
                                             eTxTobject_tval = TxTClassificationObject.eTxTobject_tval,
                                             use_pretrained_embedding = True,
                                             padding=True,
                                             maxlen = 150,
                                             nn_parameter = nn_parameter,
                                             embedding_parameter = embedding_parameter,
                                             loss = torch.nn.CrossEntropyLoss(),
                                             optimizer = 'adam')

TxTClassificationObject_2.nnSentenceClassification(metric = 'f1_score', 
                                                 save_pytorch_classifier=True,
                                                 path = 'C:/Users/Mehya/OneDrive/Desktop/enowa/NLP/src/enlp2')

report_2 = TxTClassificationObject_2.evaluate(report_classification = True, out_of_sample=True)


