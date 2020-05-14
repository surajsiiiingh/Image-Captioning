import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size=embed_size
        self.hidden=hidden_size
        self.vocab=vocab_size
        self.n_layers=num_layers
        self.embed=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(embed_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embed(captions[:,:-1])
        embeddings = torch.cat([features.unsqueeze(1), embeddings], 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.fc(hiddens)
        return outputs
          

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption=[]
        hidden = (torch.randn(self.n_layers, 1, self.hidden).to(inputs.device),
                  torch.randn(self.n_layers, 1, self.hidden).to(inputs.device))
        
        for i in range(max_len):
#             print(inputs.size())
            out,hidden=self.lstm(inputs,hidden)
            output=self.fc(out)
#             print(output)
#             print(output.size())
            capid  = torch.argmax(output,dim=2)
#             print(capid)
            caption.append(capid.item())
#             print(caption)
            
            inputs=self.embed(capid)
        return caption