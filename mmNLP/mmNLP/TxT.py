import numpy as np
import pickle 
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
    
class TxT(object):
    def __init__(self,
                 txt_lines,
                 path,
                 label = None,
                 tokenizer = None,
                 encode_label = True,
                 preprocessor = None,
                 drop_null = False,
                 min_words_in_sentence = False,
                 build_vocab = True,
                 ):
        
        """

        Parameters:
        ----------
        txt_lines : Pandas Series 
            Each line is a text string that represents a document.
        path : String
            Path for saving and loading a vocabulary
        label : Pandas Series, optional
            Each line represents the associated label for the document in txt_lines. The default is None.
        preprocessor : function, optional
            Function for preprocessing the text data. The default is None.
        ----------

        """
        
        self.txt_lines = txt_lines
        self.path = path
        self.label = label
        
        if tokenizer is None:
            self.tokenizer = word_tokenize
        
        if min_words_in_sentence:
            self.txt_lines = txt_lines.apply(self.min_words_in_sentence)
        
        if label is not None:
            if drop_null:
                self.txt_lines = self.txt_lines[(~label.isnull()) & (~self.txt_lines.isnull())  & (label != ' ') & (label != '') & (self.txt_lines != '0')]
                self.label = self.label[self.txt_lines.index]
                self.txt_lines = self.txt_lines.reset_index(drop = True)
                self.label = self.label.reset_index(drop = True)
                
            self.num_classes = len(np.unique(self.label))
            
            if encode_label:
                try:
                    file = open(self.path + '/encoder.pkl', 'rb')
                    self.encoder = pickle.load(file)
                    file.close()
                    
                    self.encoded_label = self.encoder.transform(self.label)
                except OSError:
                    self.encoder = LabelEncoder()
                    self.encoder.fit(self.label)
                    self.encoded_label = self.encoder.transform(self.label)
                    
                    file = open(self.path + '/encoder.pkl', 'wb')
                    pickle.dump(self.encoder, file)
                    file.close()
            else:
                self.encoded_label = label
            
        else:
            if drop_null:
                self.txt_lines = self.txt_lines[(~self.txt_lines.isnull()) & (self.txt_lines != '0')]
                self.txt_lines = self.txt_lines.reset_index(drop = True)
        
        if preprocessor is not None:
            self.txt_lines = self.txt_lines.apply(preprocessor)
        
        try:
            file = open(self.path + '/vocabulary.pkl','rb')
            try:
                self.vocabulary = pickle.load(file)
            except EOFError:
                file.close()
                if build_vocab:
                    self.vocabulary = self.buildVocabulary()
        except OSError:
            if build_vocab:
                self.vocabulary = self.buildVocabulary()
        
        if build_vocab:
            self.vocab_size = len(self.vocabulary)
    
    
    def to_categorical(self):
        return  np.eye(self.num_classes, dtype='uint8')[self.encoded_label]
    
    
    def buildVocabulary(self):
        vocabulary = {}
        vocabulary['<pad>'] = 0
        vocabulary['<unk>'] = 1
        index = 2
        for line in self.txt_lines:
            tokenized_line = self.tokenizer(line)

            for token in tokenized_line:
                if token not in vocabulary:
                    vocabulary[token] = index
                    index += 1
        
        file = open(self.path + '/vocabulary.pkl','wb')
        pickle.dump(vocabulary,file)
        file.close()
                    
        return vocabulary
    
    
    def min_words_in_sentence(self, txt, k=4):
        if isinstance(txt, str):
            text = self.tokenizer(txt)
            if len(text) < k:
                text = '0'
            else:
                text = txt
        else:
            text = txt
            
        return text
    
    def tokenize(self):
        return self.txt_lines.apply(self.tokenizer)
    
    
    def getSequences(self):
        
        tokenized_texts = self.tokenize()
        
        sequences = []
        for tokenized_line in tokenized_texts:
            sequence = []
            for token in tokenized_line:
                if token in self.vocabulary:
                    sequence.append(self.vocabulary.get(token))
                else:
                    sequence.append(self.vocabulary.get('<unk>'))
            sequences.append(sequence)
        
        return sequences
    
    
    def getPaddedSequences(self, 
                           maxlen = None,
                           padding='post'
                           ):
        
        if maxlen is None:
            maxlen = max(self.txt_lines.apply(len))
        
        tokenized_texts = self.tokenize()

        padded_sequences = []
        for tokenized_line in tokenized_texts:
            
            if maxlen - len(tokenized_line) < 0:
                
                if padding == 'post':
                    tokenized_line = tokenized_line[0:maxlen]
                elif padding == 'pre':
                    tokenized_line = tokenized_line[(len(tokenized_line)-maxlen):len(tokenized_line)]
                else:
                    raise NameError("Use either 'post' or 'pre' for a padding method")
            
            else:
                
                if padding == 'post':
                    tokenized_line = tokenized_line + ['<pad>'] * (maxlen - len(tokenized_line))
                elif padding == 'pre':
                    tokenized_line = ['<pad>'] * (maxlen - len(tokenized_line)) + tokenized_line
                else:
                    raise NameError("Use either 'post' or 'pre' for a padding method")
             
            padded_sequence = []
            for token in tokenized_line:
                
                if token in self.vocabulary:
                    padded_sequence.append(self.vocabulary.get(token))
                else:
                    padded_sequence.append(self.vocabulary.get('<unk>'))
                
            padded_sequences.append(padded_sequence)
        
        return padded_sequences
    
       