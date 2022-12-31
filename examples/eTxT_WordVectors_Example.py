#libraries
from mmNLP.TextModule import TxT
from mmNLP.WordVectorsModule import WordVectors
import pandas as pd
import re

# %% Laden und Vorverarbeiten der Daten
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


# %% eTxT-Object
txtobject = TxT(txt_lines = txt, 
                label = label, 
                path = 'C:/Users/Mehya/OneDrive/enowa/NLP/src/enlp4', 
                preprocessor = preproc,
                build_vocab = True)


val_counts = txtobject.label.value_counts()

vocab = txtobject.buildVocabulary()

sequences = txtobject.getSequences()

padded_sequences = txtobject.getPaddedSequences()


# %% eWordVector-Object 
embedding_parameter = {'model_type':'fasttext',
                       'embedding_size':500, 
                       'window_size':9, 
                       'subword_ngrams_min':3, 
                       'subword_ngrams_max':8}


WordVectorObject = WordVectors(TxTobject = txtobject,
                               embedding_parameter = embedding_parameter)


#hole naechste nachbarn
nearest_sentences = WordVectorObject.getNearestSentences(WordVectorObject.txt_lines[2], k_nearest=5)
print(nearest_sentences[1])


#hole naechste nachbarn
nearest_sentences = WordVectorObject.getNearestSentences('rentenbeginn', k_nearest=5)
print(nearest_sentences[1])


#hole Wordvektoren
embedding = WordVectorObject.getEmbeddings()


#hole Dokumentenvektoren
senctence_embedding = WordVectorObject.getEmbeddings(sentence_vectors=True)


#analyze label
list_of_nearest_sentences, list_of_labels, unqiue_labels = WordVectorObject.analyzeLabel(set_labels_for_reference_sentences = True)
print(list_of_nearest_sentences[6][4])


