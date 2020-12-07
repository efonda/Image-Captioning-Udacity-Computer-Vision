import torch
import torch.nn as nn
import torchvision.models as models


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # added

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
        # below
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
      
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        
        embeds = self.word_embeddings(captions[:,:-1])
        
        inputs = torch.cat((features.unsqueeze(1),embeds),1)
        
        hidden = (torch.randn(1, captions.shape[0], self.hidden_size).cuda(), 
                  torch.randn(1, captions.shape[0], self.hidden_size).cuda())
        
        lstm_out, hidden = self.lstm(inputs, hidden)
        
        out = self.fc(lstm_out)
        
        return out
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        out_sentence = []
        
        hidden = (torch.randn(1, 1, self.hidden_size).cuda(), 
                  torch.randn(1, 1, self.hidden_size).cuda())
        lstm_out, hidden = self.lstm(inputs, hidden)
        out_fc = self.fc(lstm_out)
        prediction = out_fc.argmax(dim=2)
        for i in range(max_len):         
            lstm_out, hidden = self.lstm(inputs, hidden)
            out_fc = self.fc(lstm_out)
            prediction = out_fc.argmax(dim=2)
            out_sentence.append(prediction[0].item())
            inputs = self.word_embeddings(prediction)
            
        return out_sentence