# UULearning
Code accompanying the UQ AIML x MSS [talk](https://fb.me/e/1n866e5x8): Supervised Learning without Supervision for Cyber-Attack Detection.

Download UNSW-NB15 dataset from [here](https://uq-my.sharepoint.com/:f:/g/personal/uqjwilt1_uq_edu_au/Et3QOlgWvtdGs72GZysbSfwBc7ImwQfoVzi_hYNQsBk8eg?e=hObv9N) (UQ login required).

Save data to directory ```UNSW/```.


## Requirements:
```train.py```, ```neural_network.py```, [torch](https://pytorch.org/), [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/).


## Reproducing Code
```
python train.py
```
## Using as an sklearn Classifier 
This code can also be used like an sklearn-style classifier to train a neural network classifier using either a fully labelled dataset or two unlabelled datasets. An example usage for supervised classification:
```
import pandas as pd
import numpy as np
from neural_network import PNClassifier

P_train = np.array(pd.read_csv('data/P_train.csv'))
N_train = np.array(pd.read_csv('data/N_train.csv'))
X_test = np.array(pd.read_csv('data/X_test.csv'))
y_test = np.array(pd.read_csv('data/y_test.csv'))

g = PNClassifier(hidden_layer_sizes = (300, 300, 300))
g.fit(P_train, N_train)
predictions = g.predict(X_test)
print('Accuracy', (predictions == y_test).mean())
```

An example of using the unlabelled classifier:
```
import pandas as pd
import numpy as np
from neural_network import UUClassifier

U1 = np.array(pd.read_csv('data/U1_train.csv'))
U2 = np.array(pd.read_csv('data/U2_train.csv'))
X_test = np.array(pd.read_csv('data/X_test.csv'))
y_test = np.array(pd.read_csv('data/y_test.csv'))

g = UUClassifier(hidden_layer_sizes = (300, 300, 300), pi, theta1, theta2)
g.fit(U1, U2)
predictions = g.predict(X_test)
print('Accuracy', (predictions == y_test).mean())
```

