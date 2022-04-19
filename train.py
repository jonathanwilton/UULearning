import pandas as pd
import numpy as np
from neural_network import PNClassifier, UUClassifier
import matplotlib.pyplot as plt

#%% UNSW data
X_train = np.array(pd.read_csv("UNSW/X_train.csv"))
y_train = 2*np.array(pd.read_csv("UNSW/y_train.csv")).flatten()-1
X_test = np.array(pd.read_csv("UNSW/X_test.csv"))
y_test = 2*np.array(pd.read_csv("UNSW/y_test.csv")).flatten()-1

pi = (y_train == 1).mean()

def main():
    #%% Supervised 
    P = X_train[y_train == 1]
    N = X_train[y_train == -1]
    
    g = PNClassifier(hidden_layer_sizes = (300, 300, 300, 300), 
                     pi = pi, 
                     lr = 1e-3)
    
    g.fit(P, N)
    predictions = g.predict(X_test)
    
    TP = (predictions[y_test == 1] == 1).sum().item()
    TN = (predictions[y_test == -1] == -1).sum().item()
    FP = (predictions[y_test == -1] == 1).sum().item()
    FN = (predictions[y_test == 1] == -1).sum().item()
    
    PN_accuracy = (TP+TN)/(TP+TN + FP+FN)
    PN_F = 2*TP/(2*TP+FP+FN)
    
    # plt.figure()
    # plt.plot(g.epoch_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.show()
    
    #%% UU data setup
    n_p = (y_train == 1).sum()
    n_n = (y_train == -1).sum()
    
    # choose proportions of positive data in each unlabelled dataset
    theta1 = 0.8
    theta2 = 0.2
    
    # n_1, n_2 = min((y_train == 1).sum(), (y_train == -1).sum()), min((y_train == 1).sum(), (y_train == -1).sum())
    n_1 = int(min(n_p/theta1, n_n/(1-theta1)))
    n_2 = int(min(n_p/theta2, n_n/(1-theta2)))
    
    prop_p_to_1 = theta1 * n_1/n_p
    prop_n_to_1 = (1-theta1)*n_1/n_n
    prop_p_to_2 = theta2 * n_2/n_p
    prop_n_to_2 = (1-theta2)*n_2/n_n
    
    p_to_1_ind = np.random.choice(np.where(y_train == 1)[0], size = int(prop_p_to_1*n_p), replace = False)
    p_to_2_ind = np.random.choice(np.where(y_train == 1)[0], size = int(prop_p_to_2*n_p), replace = False)
    
    n_to_1_ind = np.random.choice(np.where(y_train == -1)[0], size = int(prop_n_to_1*n_n), replace = False)
    n_to_2_ind = np.random.choice(np.where(y_train == -1)[0], size = int(prop_n_to_2*n_n), replace = False)
    
    U1 = X_train[np.concatenate((p_to_1_ind, n_to_1_ind))]
    U2 = X_train[np.concatenate((p_to_2_ind, n_to_2_ind))]
        
    #%%
    g = UUClassifier(hidden_layer_sizes = (300, 300, 300, 300), 
                     pi = pi,
                     theta1 = theta1,
                     theta2 = theta2,
                     lr = 1e-3)
    
    g.fit(U1, U2)
    predictions = g.predict(X_test)
    
    TP = (predictions[y_test == 1] == 1).sum().item()
    TN = (predictions[y_test == -1] == -1).sum().item()
    FP = (predictions[y_test == -1] == 1).sum().item()
    FN = (predictions[y_test == 1] == -1).sum().item()
    
    UU_accuracy = (TP+TN)/(TP+TN + FP+FN)
    UU_F = 2*TP/(2*TP+FP+FN)
    
    # plt.figure()
    # plt.plot(g.epoch_losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Training Loss')
    # plt.show()
    
    return PN_accuracy, PN_F, UU_accuracy, UU_F

#%% 
if __name__ == '__main__':
    replications = 5
    accuracies_PN = np.zeros([replications])
    fs_PN = np.zeros([replications])
    accuracies_UU = np.zeros([replications])
    fs_UU = np.zeros([replications])
    
    for i in range(replications):
        accuracies_PN[i], fs_PN[i], accuracies_UU[i], fs_UU[i] = main()
    
    print('\nPN accuracy mean (sd): ' + str(round(accuracies_PN.mean(), 4)) + ' (' + str(round(fs_PN.std(), 4)) + ')')
    print('PN F mean (sd): ' + str(round(fs_PN.mean(), 4)) + ' (' + str(round(fs_PN.std(), 4)) + ')')
    
    print('\nUU accuracy mean (sd): ' + str(round(accuracies_UU.mean(), 4)) + ' (' + str(round(fs_UU.std(), 4)) + ')')
    print('UU F mean (sd): ' + str(round(fs_UU.mean(), 4)) + ' (' + str(round(fs_UU.std(), 4)) + ')')
