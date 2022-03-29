# Weakly supervised binary classifiers
# try to combine all types with a super class and sub classes
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.mlp.extend([nn.Sequential(nn.Linear(sizes[i], sizes[i+1]), nn.ReLU()) for i in range(len(sizes)-2)])
        self.mlp.append(nn.Linear(sizes[-2], sizes[-1]))
        
        for i in range(len(self.mlp)):
            if i < len(self.mlp)-1:
                nn.init.kaiming_uniform_(self.mlp[i][0].weight)
    
    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class MLPClassifier():
    def __init__(self, hidden_layer_sizes = (100,), 
                 device = None,
                 loss = 'sigmoid',
                 solver = 'adam', 
                 batch_size = 1024,
                 lr = 1e-3,
                 weight_decay = 5e-8,
                 max_iter = 100,
                 tol = 0.0001,
                 n_iter_no_change = 10,
                 print_progress = False,
                 print_freq = 1,
                 validation = False,
                 val_prop = 0.1):
        
        self.hidden_layer_sizes = hidden_layer_sizes
        self.device = device
        self.loss = loss
        self.solver = solver
        self.batch_size = batch_size
        self.lr = lr
        self.wd = weight_decay
        self.max_iter = max_iter
        self.tol = tol 
        self.n_iter_no_change = n_iter_no_change
        self.print_progress = print_progress
        self.print_freq = print_freq
        self.validation = validation
        self.val_prop = val_prop
        
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    
    def l(self, yhat, y, version = 'logistic'):
        if version == 'logistic':
            out = F.softplus(-yhat*y)
        elif version == 'sigmoid':
            out = torch.sigmoid(-yhat*y)
        elif version == '01':
            out = (1-torch.sign(yhat*y))/2
        else:
            return 'not valid loss type'
        
        return out.mean()
        
    

    
    def train(self, X1, X2, training_loss):
        d = X1.shape[1]        

        self.g = MLP([d] + list(self.hidden_layer_sizes) + [1])
        self.g.to(self.device)
        
        X1 = np.array(X1)
        X2 = np.array(X2)
        
        # scale data between 0 and 1
        self.maximum = np.max(np.concatenate((X1,X2), axis = 0), axis = 0)
        self.minimum = np.min(np.concatenate((X1,X2), axis = 0), axis = 0)
        X1 = (X1 - self.minimum)/(self.maximum - self.minimum)
        X2 = (X2 - self.minimum)/(self.maximum - self.minimum) 
        
        if self.validation:
            x1_val_inds = np.random.choice(np.arange(len(X1)), size = int(self.val_prop*len(X1)), replace = False)
            x2_val_inds = np.random.choice(np.arange(len(X2)), size = int(self.val_prop*len(X2)), replace = False)        
            x1_train_inds = np.setdiff1d(np.arange(len(X1)), x1_val_inds)
            x2_train_inds = np.setdiff1d(np.arange(len(X2)), x2_val_inds)
            
            X1_val = X1[x1_val_inds]
            X2_val = X2[x2_val_inds]
            X1 = X1[x1_train_inds]
            X2 = X2[x2_train_inds]
        
        n_1 = len(X1)
        n_2 = len(X2)

        X1 = torch.Tensor(X1)
        X2 = torch.Tensor(X2)
                        
        if self.validation:
            X1_val = torch.Tensor(X1_val).to(self.device)
            X2_val = torch.Tensor(X2_val).to(self.device)
        
        # shuffle data
        X1 = X1[torch.randperm(n_1)]
        X2 = X2[torch.randperm(n_2)]
        
        self.num_batches = int((n_1 + n_2)/self.batch_size)        
        self.batch_size_1 = int(n_1/self.num_batches)
        self.batch_size_2 = int(n_2/self.num_batches)
        
        x1_batches = []
        x2_batches = []
        
        for i in range(self.num_batches):
            if i < self.num_batches - 1:
                x1_batches += [X1[i*self.batch_size_1: (i+1)*self.batch_size_1]]
                x2_batches += [X2[i*self.batch_size_2: (i+1)*self.batch_size_2]]
            else:
                x1_batches += [X1[i*self.batch_size_1:]]
                x2_batches += [X2[i*self.batch_size_2:]]
        
        if self.solver == 'adam':
            optimiser = torch.optim.Adam(self.g.parameters(), lr = self.lr, weight_decay = self.wd)
        if self.solver == 'adadelta':
            optimiser = torch.optim.Adadelta(self.g.parameters(), lr = self.lr, weight_decay = self.wd)
        if self.solver == 'sgd':
            optimiser = torch.optim.SGD(self.g.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.wd)
        
        self.losses = []
        epoch_losses = [10e20]
        locked = epoch_losses[0]
        
        if self.validation:
            epoch_val_losses = [10e20]
        
        increased_count = 0 # count number of iterations epoch loss does not decrease by at least tol
        for epoch in range(self.max_iter):            
            for i, (x1, x2) in enumerate(zip(x1_batches, x2_batches)):
                x1, x2 = x1.to(self.device), x2.to(self.device)
                optimiser.zero_grad()
                g1, g2 = self.g(x1), self.g(x2)
                loss = training_loss(g1, g2)
                loss.backward()
                optimiser.step()
                self.losses.append(loss.item())
            epoch_losses.append(loss.item())
            
            if self.validation:
                with torch.no_grad():
                    g1_val, g2_val = self.g(X1_val), self.g(X2_val)
                    validation_loss = training_loss(g1_val, g2_val)
                    epoch_val_losses += [validation_loss.item()]
                    
                if locked - epoch_val_losses[-1] < self.tol:
                    increased_count += 1
                else:
                    locked = epoch_val_losses[-1]
                    increased_count = 0
                    
            else:    
                if locked - epoch_losses[-1] < self.tol:
                    increased_count += 1
                else:
                    locked = epoch_losses[-1]
                    increased_count = 0
                
            if self.print_progress:
                if epoch % self.print_freq == 0:
                    if self.validation:
                        with torch.no_grad():
                            print('Epoch: ' + str(epoch) + '/' + str(self.max_iter) + 
                                  ', training loss = ' + str(round(epoch_losses[-1], 4)) + 
                                  ', validation loss = ' + str(round(epoch_val_losses[-1], 4)) + 
                                  ', increased_count = ' + str(increased_count))
                    else:
                        print('Epoch: ' + str(epoch) + '/' + str(self.max_iter) + 
                              ', training loss = ' + str(round(epoch_losses[-1], 4)) + 
                              ', increased_count = ' + str(increased_count))
            
            if increased_count >= self.n_iter_no_change:
                if self.print_progress:
                    print('stopped early')
                self.epochs = epoch
                break
                        
        self.epoch_losses = epoch_losses[1:]
        
        if self.validation:
            self.epoch_val_losses = epoch_val_losses
            
        return self

    
    def predict(self, X):
        X = torch.Tensor((np.array(X) - self.minimum)/(self.maximum - self.minimum)).to(self.device)
        preds = torch.zeros(len(X))
        batch_size_test = int(len(X)/self.num_batches)
        
        test_batches = []
        
        for i in range(self.num_batches):
            if i < self.num_batches - 1:
                test_batches += [X[i*batch_size_test: (i+1)*batch_size_test]]
            else:
                test_batches += [X[i*batch_size_test:]]
        
        with torch.no_grad():
            for i, x in enumerate(test_batches):
                x = x.to(self.device)
                pred = 2*(self.g.forward(x) > 0).type(torch.int8) - 1
                
                if i < self.num_batches - 1:
                    preds[i*batch_size_test: (i+1)*batch_size_test] = pred.cpu().flatten()
                else:
                    preds[i*batch_size_test:] = pred.cpu().flatten()
            
        return np.array(torch.Tensor(preds).flatten())


#%%
class PNClassifier(MLPClassifier):
    def __init__(self, hidden_layer_sizes = (100,), 
                 pi = None,
                 device = None,
                 loss = 'sigmoid',
                 solver = 'adam', 
                 batch_size = 1024,
                 lr = 1e-3,
                 weight_decay = 5e-8,
                 max_iter = 100,
                 tol = 0.0001,
                 n_iter_no_change = 10,
                 print_progress = False,
                 print_freq = 1,
                 validation = False,
                 val_prop = 0.1):
        
        self.pi = pi
        super().__init__(hidden_layer_sizes = hidden_layer_sizes, 
                         device = device,
                         loss = loss,
                         solver = solver, 
                         batch_size = batch_size,
                         lr = lr,
                         weight_decay = weight_decay,
                         max_iter = max_iter,
                         tol = tol,
                         n_iter_no_change = n_iter_no_change,
                         print_progress = print_progress,
                         print_freq = print_freq,
                         validation = validation,
                         val_prop = val_prop)
        
    def fit(self, P, N):        
        # prior
        if self.pi is None:
            self.pi = len(P)/((len(P)) + len(N))
        
        def training_loss(gp, gn):
            risk_pos = self.pi*self.l(gp, 1, self.loss)
            risk_neg = (1-self.pi)*self.l(gn, -1, self.loss)
            return risk_pos + risk_neg
                
        super().train(P, N, training_loss)
        return self
    
    def predict(self, X):
        preds = super().predict(X)
        return preds
    
#%%
class UUClassifier(MLPClassifier):
    def __init__(self, hidden_layer_sizes = (100,), 
                 pi = None,
                 theta1 = None,
                 theta2 = None,
                 device = None,
                 loss = 'sigmoid',
                 solver = 'adam', 
                 batch_size = 1024,
                 lr = 1e-3,
                 weight_decay = 5e-8,
                 max_iter = 100,
                 tol = 0.0001,
                 n_iter_no_change = 10,
                 print_progress = False,
                 print_freq = 1,
                 validation = False,
                 val_prop = 0.1):
        
        self.pi = pi
        self.theta1 = theta1
        self.theta2 = theta2
        
        self.a = (1-self.theta2)*self.pi/(self.theta1 - self.theta2)
        self.b = self.theta2*(1-self.pi)/(self.theta1 - self.theta2)
        self.c = (1-self.theta1)*self.pi/(self.theta1 - self.theta2)
        self.d = self.theta1*(1-self.pi)/(self.theta1 - self.theta2)
        
        super().__init__(hidden_layer_sizes = hidden_layer_sizes, 
                     device = device,
                     loss = loss,
                     solver = solver, 
                     batch_size = batch_size,
                     lr = lr,
                     weight_decay = weight_decay,
                     max_iter = max_iter,
                     tol = tol,
                     n_iter_no_change = n_iter_no_change,
                     print_progress = print_progress,
                     print_freq = print_freq,
                     validation = validation,
                     val_prop = val_prop)
    
        
    def fit(self, U1, U2):
        # prior
        if (self.pi is None) or (self.theta1 is None) or (self.theta2 is None):
            print('Please specify all of pi, theta1 and theta2')
        
        def training_loss(g1, g2):            
            risk1 = torch.abs(self.a*self.l(g1, 1, self.loss) - self.c*self.l(g2, 1, self.loss))
            risk2 = torch.abs(self.d*self.l(g2, -1, self.loss) - self.b*self.l(g1, -1, self.loss))
            return risk1 + risk2
        
        super().train(U1, U2, training_loss)
        return self
    
    def predict(self, X):
        preds = super().predict(X)
        return preds