from torch import nn
import torch.nn.functional as F
import time

class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        
        return F.log_softmax(self.linear(bow_vec), dim=1)
    





def train_an_epoch(dataloader, model, optimizer, loss_fn, verbose = True):
    
    
    model.train() 
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        log_probs = model(text)
        loss = loss_function(log_probs, label)
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0 and idx > 0 and verbose:
            print(f'At iteration {idx} the loss is {loss:.3f}.')

def train_BOW(dataloader, 
                model, 
                verbose = True, 
                loss_fn = nn.NLLLoss(),
                optimizer = "adam",
                learning_rate = 0.01):

    optimizer = nn.optim.Adam(model.parameters(), lr=0.001)
    
