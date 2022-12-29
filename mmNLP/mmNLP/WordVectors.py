import numpy as np
import pandas as pd
import csv

#sklearn
from sklearn.cluster import KMeans

#modules for wordvectors
from fasttext import FastText
from scipy.spatial import distance

#Network architectures
from . import TxT

class WordVectors(object):
    def __init__(self,
                 txt_lines = None,
                 path = None,
                 label = None,
                 TxTobject = None,
                 preprocessor = None,
                 embedding_parameter = None,
                 distance_func = distance.cosine,
                 sentence_vector = True,
                 save_model = False,
                 ):
        
        
        if TxTobject is None:
            self.TxTobject = TxT(txt_lines = txt_lines, 
                                 path = path, 
                                 label = label, 
                                 preprocessor = preprocessor)
        else:
            self.TxTobject = TxTobject
        
        self.txt_lines = self.TxTobject.txt_lines.reset_index(drop=True)
        self.path = self.TxTobject.path
        self.label = self.TxTobject.label.reset_index(drop=True)
        self.embedding_parameter = embedding_parameter
        self.label_dictionary = self.buildLabelDictionary()
        self.distance_func = distance_func
        self.sentence_vector = sentence_vector
        
        if self.embedding_parameter is None:
            self.embedding_parameter = {'model_type':'fasttext',
                                        'embedding_size':300, 
                                        'window_size':6, 
                                        'subword_ngrams_min':3, 
                                        'subword_ngrams_max':6}
        
        
        if self.embedding_parameter['model_type'] == 'fasttext':
            self.saveFasttextFile()
            self.word_model = FastText.train_unsupervised(self.path + '/textfile_for_fasttext.txt',
                                                          dim=embedding_parameter['embedding_size'],
                                                          ws=embedding_parameter['window_size'],
                                                          minn=embedding_parameter['subword_ngrams_min'],
                                                          maxn=embedding_parameter['subword_ngrams_max'])
            
            if save_model:
                self.word_model.save_model(self.path + '/fasttext.bin')
    
    def __compute_distance(self,
                           txt1, 
                           txt2):
        
        if self.embedding_parameter['model_type'] == 'fasttext':
            if self.sentence_vector == False:
                v1 = self.word_model.get_word_vector(txt1)
                v2 = self.word_model.get_word_vector(txt2)
            
            else:
                v1 = self.word_model.get_sentence_vector(txt1)
                v2 = self.word_model.get_sentence_vector(txt2)
        
        return self.distance_func(v1,v2)
        
            
    def buildDistanceDictionary(self):
        if self.embedding_parameter['model_type'] == 'fasttext':
            distance_dict = {}
            for i in len(self.txt_lines):
                line_distance_dict = {}
                for j in len(self.txt_lines):
                    line_distance_dict[self.txt_lines[j]] = self.__compute_distance(self.txt_lines[i],self.txt_lines[j])
                
                distance_dict[self.txt_lines[i]] = line_distance_dict
             
                
    def saveFasttextFile(self):
        self.txt_lines.to_csv(self.path + '/textfile_for_fasttext.txt', 
                              sep = ' ', 
                              index=False, 
                              header=None,
                              quoting = csv.QUOTE_NONE,
                              quotechar = "",
                              escapechar = " ")
        
    
    def buildLabelDictionary(self):
        label_dictionary = {}
        for i in range(len(self.txt_lines)):
            label_dictionary[self.txt_lines[i]] = self.label[i] 
            
        return label_dictionary
    
    
    def buildSentenceDictionary(self,
                                reference_senctence,
                                load = False
                                ):
        
        if self.embedding_parameter['model_type'] == 'fasttext': 
            
            if load:
               self.word_model = FastText.load_model(self.path + '/fasttext.bin')
                   
            sentence_dictionary = {}
            reference_senctence_vector = self.word_model.get_sentence_vector(reference_senctence)
            
            for i in range(len(self.txt_lines)):
                dist = self.distance_func(reference_senctence_vector,self.word_model.get_sentence_vector(self.txt_lines[i]))
                sentence_dictionary[self.txt_lines[i]] = dist 

        else:
            raise ValueError("This method is only available for 'fasttext'")
            
        return sentence_dictionary
    
    
    def getSentenceVector(self,txt):
        
        tokenized_line = self.tokenizer(txt)
        np_word_matrix = np.array(self.getEmbeddings(list_of_words=tokenized_line))
        
        for i in range(len(np_word_matrix)):
            x = np_word_matrix[i]
            norm = np.sqrt(np.sum(x**2))
            if i == 0:
                if norm > 0: 
                    summ = x*(1.0/norm)
                else:
                    summ = x
            else:
                if norm > 0: 
                    summ += x*(1.0/norm)
                else:
                    summ += x 
                    
        return summ 
    
    
    def sentenceClustering(self,
                           method = 'Kmeans',
                           n_clusters = None,
                           plot_3d = False,
                           dimension_reduction = 'PCA'
                           ):
        
        
        sentence_vectors = pd.DataFrame(self.getEmbeddings(sentence_vectors = True))
       
        if method == 'Kmeans':
           
            if n_clusters is None:
                n_clusters = self.TxTobject.num_classes
           
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(sentence_vectors)
            sentence_vectors['Cluster'] = kmeans.predict(sentence_vectors)
           
            return sentence_vectors
            
    def getNearestSentences(self, 
                            reference_senctence,
                            sentence_dictionary = None,
                            k_nearest = 3,
                            distance_func = distance.cosine, 
                            model = 'fasttext'
                            ):
        
        if sentence_dictionary is None:
            sentence_dictionary = self.buildSentenceDictionary(reference_senctence)
            
        return pd.Series(sorted(sentence_dictionary, key=sentence_dictionary.get, reverse=False)[:k_nearest])
    
    
    def analyzeLabel(self, 
                     set_labels_for_reference_sentences = False,
                     distance_func = distance.cosine,
                     null_class = 'Irrelevant'
                     ):
        
        if self.label is not None:
            list_of_nearest_sentences = []
            unqiue_labels = self.label.unique()
            for i in unqiue_labels:
                for j in range(len(self.label)):
                    if self.label[j] == i:
                        if self.label[j] != null_class:
                            reference_senctence = self.txt_lines[j]
                            label_counts = self.label.value_counts()[self.label[j]]
                            sentence_dictionary = self.buildSentenceDictionary(reference_senctence)
                            
                            if set_labels_for_reference_sentences:
                                reference_senctence = self.label[j]
                                
                            nearest_sentences = self.getNearestSentences(reference_senctence = reference_senctence,
                                                                         sentence_dictionary = sentence_dictionary,
                                                                         k_nearest = label_counts,
                                                                         distance_func = distance_func)
                            
                            list_of_nearest_sentences.append(nearest_sentences)
                        
                            break
            
            list_of_labels = []
            for nearest_sentences in list_of_nearest_sentences:
                labels = []
                for t in range(len(nearest_sentences)):
                    labels.append(self.label_dictionary.get(nearest_sentences[t]))
                
                list_of_labels.append(labels)
                
        
        return list_of_nearest_sentences, list_of_labels, unqiue_labels

    def getEmbeddings(self, 
                     list_of_words = None, 
                     list_of_sentences = None,
                     sentence_vectors = False, 
                     ):
        
        if list_of_words is None:
            list_of_words = self.TxTobject.vocabulary.copy()
            
        if list_of_sentences is None:
            list_of_sentences = self.txt_lines.copy()
        
        list_of_sentences = list_of_sentences.reset_index(drop=True)
        
        if self.word_model is not None:
            if self.embedding_parameter['model_type'] == 'fasttext':
                embedding_size = self.word_model.get_dimension()
                if sentence_vectors:
                    embedding_matrix = np.zeros(shape=(len(list_of_sentences),embedding_size))
                    for i in range(len(list_of_sentences)):
                        embedding_matrix[i] = self.word_model.get_sentence_vector(list_of_sentences[i])
                else:
                    embedding_matrix = np.random.uniform(-0.4, 0.4, (len(list_of_words), embedding_size))
                    embedding_matrix[self.TxTobject.vocabulary['<pad>']] = np.zeros((embedding_size,))
                    for word in list_of_words:
                        if word in self.TxTobject.vocabulary:
                            embedding_matrix[self.TxTobject.vocabulary[word]] = self.word_model.get_word_vector(word)
        else:
            raise TypeError('To get embeddings, a Wordvector model must first be defined')
            
        return embedding_matrix