import numpy as np
import pickle

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


#torch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F

#Network architectures
from mmNLP.TextModule import TxT
from mmNLP.WordVectorsModule import WordVectors
from mmNLP.Architectures import TxTArchitectures

#Xgboost
import xgboost as xgb
    
class TxTClassification(object):
    def __init__(self,
                 txt_lines = None,
                 label = None,
                 path = None,
                 TxTobject = None,
                 TxTobject_train = None,
                 TxTobject_test = None,
                 TxTobject_tval = None,
                 nnClass = None,
                 load_pytorch_classifier = False,
                 loss = None,
                 optimizer = None,
                 use_pretrained_embedding = False,
                 embedding_parameter = None,
                 nn_parameter = None, 
                 preprocessor = None,
                 padding = True,
                 maxlen = None,
                 binary_classification = False,
                 device = torch.device("cpu")
                 ):
        
        if TxTobject is None and txt_lines is None:
            raise TypeError('An TxTobject or a text-dataset is required')
        
        if TxTobject is None:
            self.TxTobject = TxT(txt_lines = txt_lines, 
                                 path = path, 
                                 label = label, 
                                 preprocessor = preprocessor,
                                 build_vocab=False)
        else:
            self.TxTobject = TxTobject
        
        self.txt_lines = self.TxTobject.txt_lines
        self.label = self.TxTobject.label
        self.encoded_label = self.TxTobject.encoded_label
        self.num_classes = self.TxTobject.num_classes
        self.path = self.TxTobject.path
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.nn_parameter = nn_parameter
        self.classification_method = None
        self.preprocessor = preprocessor 
        self.use_pretrained_embedding = use_pretrained_embedding
        self.vectorizer = None
        self.maxlen = maxlen
        
        if TxTobject_train is None or TxTobject_test is None or TxTobject_tval is None:
            self.train, self.test, self.y_train, self.y_test = train_test_split(self.txt_lines, self.label, test_size=0.2, random_state=42)
            self.train, self.tval, self.y_train, self.y_val = train_test_split(self.train, self.y_train, test_size=0.25, random_state=42)
            
            self.TxTobject_train = TxT(self.train, path = self.path, label = self.y_train, min_words_in_sentence=False)
            self.TxTobject_test = TxT(self.test, path = self.path, label = self.y_test, min_words_in_sentence=False)
            self.TxTobject_tval = TxT(self.tval, path = self.path, label = self.y_val, min_words_in_sentence=False)
        else:
            self.TxTobject_train = TxTobject_train
            self.TxTobject_test = TxTobject_test
            self.TxTobject_tval = TxTobject_tval
            
            self.train, self.test, self.tval = TxTobject_train.txt_lines, TxTobject_test.txt_lines, TxTobject_tval.txt_lines

        self.y_train = self.TxTobject_train.encoded_label
        self.y_test = self.TxTobject_test.encoded_label
        self.y_val = self.TxTobject_tval.encoded_label
        
        if self.use_pretrained_embedding:
            if load_pytorch_classifier == False:
                self.WordVectorsObject = WordVectors(TxTobject = self.TxTobject_train,
                                                     embedding_parameter = embedding_parameter)
                
                if self.nn_parameter is not None:
                    if self.nn_parameter['embedding_size'] != self.WordVectorsObject.embedding_parameter['embedding_size']:
                        raise TypeError('embedding size from nn_parameter and embedding_parameter must be equal')
                
                self.embedding = self.WordVectorsObject.getEmbeddings()
            else:
                self.WordVectorsObject = None
                self.embedding = None
        else:
            self.WordVectorsObject = None
            self.embedding = None
            
        
        if self.nn_parameter is not None or nnClass is not None:
            self.tensor_type = torch.LongTensor
            if self.device == torch.device("cuda"):
                self.tensor_type = torch.cuda.LongTensor
            
            if self.embedding is not None:
                self.embedding = torch.tensor(self.embedding)
            
            if nnClass is None:
                self.classifier = TxTArchitectures(vocab_size = self.TxTobject_train.vocab_size,
                                                   nn_parameter = self.nn_parameter,
                                                   num_classes = self.num_classes,
                                                   pretrained_embedding = self.embedding,
                                                   method = nn_parameter['architecture'])
            else:
                self.classifier = nnClass
            
            if load_pytorch_classifier:
                self.classifier.load_state_dict(torch.load(self.path + '/classifier_parameters.pt'))
                if self.optimizer is None or self.optimizer == 'adam':
                    self.optimizer = optim.Adam(self.classifier.parameters(),
                                                lr=self.nn_parameter['learning_rate'])
                    
                self.optimizer.load_state_dict(torch.load(self.path + '/optimizer_parameters.pt'))
                self.classifier.eval()
            
            self.__train_dataloader = None
            self.__tval_dataloader = None
            self.__test_dataloader = None
            
        if padding:
            self.train = self.TxTobject_train.getPaddedSequences(maxlen = self.maxlen)
            self.test = self.TxTobject_test.getPaddedSequences(maxlen = self.maxlen)
            self.tval = self.TxTobject_tval.getPaddedSequences(maxlen = self.maxlen)
    
    
    def __transform_to_0_1(self, columm, label):
        columm[columm != label] = '0'
        columm[columm == label] = '1'
        columm = columm.astype(int)
        return columm
        
    
    def getBinaryLabels(self, null_class = 'Irrelevant'):
        binary_labels_train = []
        binary_labels_tval = []
        binary_labels_test = []
        unqiue_multiple_labels = self.label.unique()
        for i in range(self.num_classes):
            if unqiue_multiple_labels[i] != null_class:
                temp_label_train = self.TxTobject_train.label.copy()
                temp_label_tval = self.TxTobject_tval.label.copy()
                temp_label_test = self.TxTobject_test.label.copy()
                
                temp_label_train = self.__transform_to_0_1(temp_label_train, unqiue_multiple_labels[i])
                temp_label_tval = self.__transform_to_0_1(temp_label_tval, unqiue_multiple_labels[i])
                temp_label_test = self.__transform_to_0_1(temp_label_test, unqiue_multiple_labels[i])

                binary_labels_train.append(temp_label_train)
                binary_labels_tval.append(temp_label_tval)
                binary_labels_test.append(temp_label_test)
                
        return binary_labels_train, binary_labels_tval, binary_labels_test, unqiue_multiple_labels
    
    
    def torchDataLoader(self): 
        
        train = torch.tensor(self.train)
        tval = torch.tensor(self.tval)
        test = torch.tensor(self.test)
        
        y_train = torch.tensor(self.y_train).type(self.tensor_type)
        y_val = torch.tensor(self.y_val).type(self.tensor_type)
        y_test = torch.tensor(self.y_test).type(self.tensor_type)
    
        train = TensorDataset(train, y_train)
        self.__train_dataloader = DataLoader(train, batch_size=self.nn_parameter['batch_size'], shuffle=True)
        
        tval = TensorDataset(tval, y_val)
        self.__tval_dataloader = DataLoader(tval, batch_size=self.nn_parameter['batch_size'], shuffle=False)
        
        test = TensorDataset(test, y_test)
        self.__test_dataloader = DataLoader(tval, batch_size=self.nn_parameter['batch_size'], shuffle=False)
    
    
    def __print_nn(self, metric = 'accuracy'):
        
        if metric == 'f1_score':
            print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val f1-score':^9}")
            
        elif metric == 'precision':
            print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val precision':^9}")
            
        elif metric == 'recall':
            print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val recall':^9}")
            
        else:
            print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9}")
    
    
    def __compute_score(self, y_true, preds, metric = 'accuracy'):
        
        if metric == 'f1_score': 
           score = f1_score(y_true, preds, average="macro")
           
        elif metric == 'precision':
            score = precision_score(y_true, preds, average="macro")
            
        elif metric == 'recall':
           score = recall_score(y_true, preds, average="macro")
           
        else:
            score = accuracy_score(y_true, preds)
        
        return score
    
    
    def nnTrainTorch(self,metric = 'accuracy'):
        
        self.__print_nn(metric = metric)    
        print("-"*50)
        
        if self.optimizer is None or self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.classifier.parameters(),
                                        lr=self.nn_parameter['learning_rate'],
                                        amsgrad = True)

        if self.loss is None:
            self.loss = torch.nn.CrossEntropyLoss()

        
        for epoch in range(self.nn_parameter['epochs']):
            self.classifier.train()
            total_loss = []
            for step, batch in enumerate(self.__train_dataloader):
                batch_sequences, batch_labels = tuple(b.to(self.device) for b in batch)
                self.optimizer.zero_grad()
                batch_logits = self.classifier(batch_sequences)
                temporary_loss = self.loss(batch_logits, batch_labels)
                total_loss.append(temporary_loss.item())
                temporary_loss.backward()
                self.optimizer.step()
                
            avg_train_loss = np.mean(total_loss)
            val_loss, val_score = self.evaluate(metric = metric)
            print()
            print(f"{epoch + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_score:^9.2f}")

      
    def evaluate(self,
                 metric = None,
                 report_classification = False,
                 out_of_sample=False):
        
        if self.classification_method is None: 
        
            if out_of_sample:
                if self.__test_dataloader is None:
                    self.torchDataLoader()
                
                dataloader = self.__test_dataloader
                
            else:
                dataloader = self.__tval_dataloader
            
            e_loss = []
            outputs = np.array([])
            labels = np.array([])
            self.classifier.eval()
            
            for batch in dataloader:
                batch_sequences, batch_labels = tuple(b.to(self.device) for b in batch)
                
                with torch.no_grad():
                    batch_output = self.classifier(batch_sequences)
                    
                    temporary_loss = self.loss(batch_output, batch_labels)
                    e_loss.append(temporary_loss.item())
                    
                batch_labels = batch_labels.numpy()
                batch_output = np.argmax(batch_output.detach().numpy(),axis=1)
                     
                outputs = np.r_[outputs,batch_output]
                labels = np.r_[labels,batch_labels]
            
            if report_classification:
                return classification_report(labels, outputs)
            
            score = self.__compute_score(labels,outputs,metric=metric)     
            e_loss = np.mean(e_loss)
            
            return e_loss, score
        
        else:
            
            if out_of_sample:
                txt_lines = self.test
                y_true = self.y_test
            else:
                txt_lines = self.tval
                y_true = self.y_val
                
            y_pred = self.predict(txt_lines=txt_lines)
            
            if report_classification:
                return classification_report(y_true, y_pred)
            
            score = self.__compute_score(y_true,y_pred,metric=metric)

            return score
            
    
    def predict(self, 
                txt_lines,
                probabilities = False
                ):
        
        if self.nn_parameter is not None:
            TxTobject = TxT(txt_lines=txt_lines,
                            path = self.path,
                            preprocessor=self.preprocessor)
            
            dataset = torch.tensor(TxTobject.getPaddedSequences())
            logits = self.classifier.forward(dataset)
            probs = (F.softmax(logits, dim=1)).detach().numpy()
            
            if probabilities:
                return probs
            
            return np.argmax(probs, axis=1)
        
        elif self.classification_method == 'xgboost':
            
            if self.use_pretrained_embedding:
                txt_lines = self.WordVectorsObject.getEmbeddings(list_of_sentences = txt_lines,
                                                                  sentence_vectors=True)
                
            else:
                txt_lines = self.vectorizer.transform(txt_lines)
            
            return self.classifier.predict(txt_lines)
            
    
    def nnSentenceClassification(self, 
                                 metric = 'accuracy',
                                 save_pytorch_classifier = True,
                                 path = None,
                                 ):
        
        self.torchDataLoader()

        self.classifier.to(self.device)
            
        self.nnTrainTorch(metric = metric)
        
        if save_pytorch_classifier:
            
            if path is None:
                path = self.path
                
            torch.save(self.classifier.state_dict(), path + '/classifier_parameters.pt')
            torch.save(self.optimizer.state_dict(), path + '/optimizer_parameters.pt')
        

    def sentenceClassification(self,
                               classification_method = 'xgboost',
                               ngram_range = (1,1),
                               path = None,
                               ):
        
        if path is None:
            path = self.path
        
        self.classification_method = classification_method
            
        try:
            file = open(path + '/vectorizer.pkl','rb')
            self.vectorizer = pickle.load(file)
            file.close()
            
        except OSError:
            self.vectorizer = self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
            
            file = open(path + '/vectorizer.pkl', 'wb')
            pickle.dump(self.vectorizer, file)
            file.close()
        
        train = self.vectorizer.fit_transform(self.train)
        tval = self.vectorizer.transform(self.tval)
        
        eval_set = [(train,self.y_train), (tval,self.y_val)]
        
        if classification_method == 'xgboost':
            self.classifier = xgb.XGBClassifier(booster = 'dart',
                                                objective="multi:softmax", 
                                                random_state=42, 
                                                eval_metric = 'mlogloss',
                                                use_label_encoder=False)
            
            self.classifier.fit(train, 
                                self.y_train,
                                early_stopping_rounds=10,
                                eval_metric="mlogloss", 
                                eval_set=eval_set, 
                                verbose=True)
            