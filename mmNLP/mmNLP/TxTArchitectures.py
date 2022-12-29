import torch
import numpy as np
import torch.nn.functional as F

class TxTArchitectures(torch.nn.Module):
    def __init__(self, 
                 vocab_size,
                 nn_parameter,
                 num_classes,
                 pretrained_embedding = None,
                 method = 'Kim2014'
                 ):
        
        super(TxTArchitectures, self).__init__()  
        
        self.num_classes = num_classes
        self.method = method
        
        if pretrained_embedding is not None: 
            self.vocab_size = vocab_size
            self.embedding_size = nn_parameter['embedding_size']
            self.embedding = torch.nn.Embedding.from_pretrained(pretrained_embedding,
                                                                freeze=(not nn_parameter['train_embedding']))
        else:
            self.vocab_size = vocab_size
            self.embedding_size = nn_parameter['embedding_size']
            self.embedding = torch.nn.Embedding(num_embeddings=vocab_size,
                                                embedding_dim = self.embedding_size,
                                                padding_idx=0)
            
            
           
        if self.method ==  'Kim2014':
            self.conv1d_list = torch.nn.ModuleList([torch.nn.Conv1d(in_channels=self.embedding_size,
                                                                    out_channels=nn_parameter['num_filters'][i],
                                                                    kernel_size=nn_parameter['filter_sizes'][i])
                                                    for i in range(len(nn_parameter['filter_sizes']))])
            
            self.fully_connected = torch.nn.Linear(np.sum(nn_parameter['num_filters']), self.num_classes)
            self.dropout = torch.nn.Dropout(p=nn_parameter['dropout'])
        
        elif self.method == 'SimpleCNN':
            self.conv1d = torch.nn.Conv1d(in_channels=self.embedding_size,
                                          out_channels = nn_parameter['num_filters'],
                                          kernel_size = nn_parameter['filter_sizes'])
            
            self.fully_connected = torch.nn.Linear(nn_parameter['num_filters'], self.num_classes)
            self.dropout = torch.nn.Dropout(p=nn_parameter['dropout'])
        
            
    def forward(self, padded_sequences):
        if self.method ==  'Kim2014':
            x_embed = self.embedding(padded_sequences).float()
            x_reshaped = x_embed.permute(0, 2, 1)
            x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]
            x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]
            x_fully_connected = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],dim=1)
            logits = self.fully_connected(self.dropout(x_fully_connected))
            
        elif self.method == 'SimpleCNN':
            x_embed = self.embedding(padded_sequences).float()
            x_reshaped = x_embed.permute(0, 2, 1)
            x_dropout = self.dropout(x_reshaped)
            x_conv = F.relu(self.conv1d(x_dropout))
            x_pool = F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
            logits = self.fully_connected(x_pool.squeeze(dim=2))
            
        return logits
        
      