from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
from torch import nn
import torch
import copy

# load data and extract data we want to use
iris_data = datasets.load_iris()

X = iris_data.data[:100, [1, 2]]
y = iris_data.target[:100]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

print(f'X_train len:{len(X_train)}')
print(f'y_train len: {len(y_train)}')
print(f'X_test len: {len(X_test)}')
print(f'y_test len: {len(y_test)}')

# standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert to tensors
X_train_std = torch.from_numpy(X_train).type(torch.float32)
X_test_std = torch.from_numpy(X_test).type(torch.float32)
y_train = torch.from_numpy(y_train).type(torch.long)
y_test = torch.from_numpy(y_test).type(torch.long)

# make model
class NN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_shape,
                                 out_features=output_shape)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_1(self.relu(x))

model_1 = NN(input_shape=X_train.shape[1],
              output_shape=2)

# set up optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_1.parameters(),
                            lr=0.1,
                            weight_decay=.01)

torch.manual_seed(42)

epochs = 2
train_loss, test_loss = 0, 0
for epoch in range(epochs):
    model_1.train()
    y_pred = model_1(X_train_std)
    loss = loss_fn(y_pred, y_train)
    train_loss += loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # test
    model_1.eval()
    with torch.inference_mode():
        test_pred = model_1(X_test_std)
        loss_ = loss_fn(test_pred, y_test)
        test_loss += loss_

    print(f'Loss: {loss:.4f} | Test loss: {test_loss:.4f}')

# Let's try and implement a GridSearchCV with this dataset
class WrapperClass(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, output_shape, lr, epochs):
        self.input_shape=input_shape
        self.output_shape=output_shape
        self.lr=lr
        self.epochs=epochs
        self.model=None

    def fit(self, X, y):
        X = torch.Tensor(X)
        y = torch.Tensor(y)

        # define model
        model = NN(input_shape=2, output_shape=2)

        # define loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=0.01)

        # train model
        for epoch in range(self.epochs):
            train_loss = self.train_step(model, X, y, loss_fn, optimizer)
            print(f'Epoch: {epoch+1}/{self.epochs}, Loss: {train_loss:.4f}')

        self.model = model

    # get predictions 
    def predict(self, X):
        X = torch.Tensor(X)

        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model(X)
            # convert preds to class labels
            _, predicted_labels = torch.max(y_pred, axis=1)
            predicted_labels = predicted_labels.cpu().numpy()

        return predicted_labels

    # set params 
    def set_params(self, **params):
        if 'input_shape' in params:
            self.input_shape=params['input_shape']
        if 'output_shape' in params:
            self.output_shape=params['output_shape']
        if 'lr' in params:
            self.lr=params['lr']
        if 'epochs' in params:
            self.epochs=params['epochs']
        return self

    # get params
    def get_params(self, deep=True):
        return {
                'input_shape': self.input_shape,
                'output_shape': self.output_shape,
                'lr': self.lr,
                'epochs': self.epochs
                }

    # return model loss
    def train_step(self, model, X, y, loss, optimizer):
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

# create param grid to use for GridSearchCV
param_grid = {
        'input_shape': [2],
        'output_shape': [2],
        'lr': [0.0001, 0.001, 0.01, 0.1],
        'epochs': [5, 10, 15, 20, 15, 30, 50, 75, 100, 150, 200, 250, 300],
        }

wc = WrapperClass(input_shape=2,
                  output_shape=2,
                  lr=0.0001,
                  epochs=5)

# make deep copy to pass to GS 
wc_copy = copy.deepcopy(wc)

# make GridSearchCV
gc = GridSearchCV(estimator=wc_copy,
                  param_grid=param_grid,
                  cv=10,
                  scoring='accuracy')

# # train model and print best model

gc.fit(X_train_std, y_train)
print(f'Best params: {gc.best_params_}')
print(f'best score: {gc.best_score_}')

best_model = gc.best_estimator_
print(f'best model: {best_model}')



