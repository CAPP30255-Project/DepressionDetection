from torch import nn
import torch
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import torch.optim as optim

class BoWClassifier(nn.Module):

    def __init__(self, vocab_size, num_labels = 2):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)
        
    def forward(self, bow_vec):
        
        return F.log_softmax(self.linear(bow_vec), dim=1)
    





def train_an_epoch(dataloader, model, optimizer, loss_fn, using_GPU, verbose = True):
    
    
    model.train() 
    log_interval = 500

    for idx, (label, text) in enumerate(dataloader):
        model.zero_grad()
        log_probs = model(text)
        loss = loss_fn(log_probs, label)
        loss.backward()
        optimizer.step()
        if idx % log_interval == 0 and idx > 0 and verbose:
            print(f'At iteration {idx} the loss is {loss:.3f}.')

def train_BOW(data_object, 
                verbose = True, 
                loss_fn = nn.NLLLoss(),
                optimizer = "adam",
                learning_rate = 0.,
                epochs = 16,
                using_GPU = True,
                glove = False):

    model = BoWClassifier(vocab_size = 138543)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()
    if using_GPU:
        model.cuda()
        loss_function.to('cuda')
    accuracies=[]
    for epoch in range(1, epochs + 1):
        if glove:
            train_an_epoch(dataloader = data_object.bow_train_glove,
                        model = model,
                        optimizer = optimizer, 
                        loss_fn=loss_function,
                        using_GPU = using_GPU)
            accuracy = get_accuracy(data_object.bow_val_glove, model)
        else:
            train_an_epoch(dataloader = data_object.bow_train_dl,
                            model = model,
                            optimizer = optimizer, 
                            loss_fn=loss_function,
                            using_GPU = using_GPU)
            accuracy = get_accuracy(data_object.bow_val_dl, model)
        
        accuracies.append(accuracy)
        print()
        print(f'After epoch {epoch} the validation accuracy is {accuracy:.3f}.')
        print()
        
    plt.plot(range(1, epochs+1), accuracies)

def get_accuracy(dataloader, model):
    model.eval()
    with torch.no_grad():
        
        for idx, (label, text) in enumerate(dataloader):
            model.zero_grad()
            log_probs = model(text)
            _, preds = torch.max(log_probs, 1)
            correct = sum(preds == label).sum().item()
            total = len(label)
            return 100 * correct / total
