# UULearning
Code accompanying the UQ AIML x MSS [talk](https://fb.me/e/1n866e5x8): Supervised Learning without Supervision for Cyber-Attack Detection.

Download UNSW-NB15 dataset from [here](https://uq-my.sharepoint.com/:f:/g/personal/uqjwilt1_uq_edu_au/Et3QOlgWvtdGs72GZysbSfwBc7ImwQfoVzi_hYNQsBk8eg?e=hObv9N) (UQ login required).

Save data to directory ```UNSW/```.


## Requirements:
[torch](https://pytorch.org/) 

[numpy](https://numpy.org/)

[pandas](https://pandas.pydata.org/) 


## Reproducing Code
```
python train.py
```
<!-- ## Using as an sklearn Classifier 
This code can also be used like an sklearn-style classifier to train a neural network classifier using either a fully labelled dataset or two unlabelled datasets. An example usage:
```
import pandas as pd
import numpy as np
from neural_network import PNClassifier

P_train = pd.read_csv('data/P_train.csv')
N_train = pd.read_csv('data/P_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

g = PNClassifier()
g.fit(P_train, N_train)
predictions = g.predict(X_test)
print('Accuracy', (predictions == y_test).mean())
```


```
import pandas as pd
import numpy as np
from neural_network import UUClassifier

U1 = pd.read_csv('data/U1_train.csv')
U2 = pd.read_csv('data/U2_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

g = UUClassifier(pi, theta1, theta2)
g.fit(U1, U2)
predictions = g.predict(X_test)
print('Accuracy', (predictions == y_test).mean())
```
-->
