import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Config import hyperparameters
from HumBugDB.ResNetDropoutSource import resnet50dropout
from HumBugDB.ResNetSource import resnet50

class LogisticRegression:

    def __init__(self):
        """初始化Linear Regression模型"""
        self.coef_ = None
        self.intercept_ = None
        self._theta = None

    def _sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def fit(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集X_train, y_train, 使用梯度下降法训练Logistic Regression模型"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        def J(theta, X_b, y):
            y_hat = self._sigmoid(X_b.dot(theta))
            try:
                return - np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) / len(y)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            return X_b.T.dot(self._sigmoid(X_b.dot(theta)) - y) / len(X_b)

        def gradient_descent(X_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):

            theta = initial_theta
            cur_iter = 0

            while cur_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break

                cur_iter += 1

            return theta

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y_train, initial_theta, eta, n_iters)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self



    def predict_proba(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果概率向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return self._sigmoid(X_b.dot(self._theta))

    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示X_predict的结果向量"""
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        proba = self.predict_proba(X_predict)
        return np.array(proba >= 0.5, dtype='int')

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "LogisticRegression()"


class ResnetFull(nn.Module):
    def __init__(self, n_classes):
        super(ResnetFull, self).__init__()
        self.resnet = resnet50(pretrained=hyperparameters.pretrained)
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # 添加降维层
        self.dim_reduction = nn.Linear(2048, 32)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.dim_reduction(x)  # 降维到64维
        x = self.fc1(x)
        return x


class ResnetDropoutFull(nn.Module):
    def __init__(self, n_classes, dropout=0.2):
        super(ResnetDropoutFull, self).__init__()
        self.dropout = dropout
        self.resnet = resnet50dropout(
            pretrained=hyperparameters.pretrained, dropout_p=self.dropout
        )
    
        self.n_channels = 3
        # Remove final linear layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
        # 添加降维层
        self.dim_reduction = nn.Linear(2048, 32)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.resnet(x).squeeze()
        x = self.dim_reduction(x)  # 降维到64维
        x = self.fc1(F.dropout(x, p=self.dropout))
        return x


def build_dataloader(x_train, y_train, x_val=None, y_val=None, shuffle=True):
    x_train = torch.tensor(x_train)
    y_train = torch.tensor(y_train)
    print(x_train.shape)
    print(y_train.shape)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
    )
    # .unsqueeze(0)
    if x_val is not None:
        val_dataset = TensorDataset(x_val, y_val)
        val_loader = DataLoader(
            val_dataset, batch_size=hyperparameters.batch_size, shuffle=shuffle
        )

        return train_loader, val_loader
    return train_loader


def load_model(filepath, model=ResnetDropoutFull(hyperparameters.n_classes)):
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )
    model = model.to(device)

    # 直接使用device作为map_location
    print(f"Loading model from: {filepath}")
    model = torch.jit.load(filepath, map_location=device)
    model.eval()

    return model


def train_model(
    x_train,
    y_train,
    clas_weight=None,
    x_val=None,
    y_val=None,
    model=ResnetDropoutFull(hyperparameters.n_classes),
    model_name="test",
    model_dir="models",
):
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    if x_val is not None:
        train_loader, val_loader = build_dataloader(x_train, y_train, x_val, y_val)

    else:
        train_loader = build_dataloader(x_train, y_train)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using data parallel")
        model = nn.DataParallel(
            model, device_ids=list(range(torch.cuda.device_count()))
        )

    model = model.to(device)

    if clas_weight is not None:
        print("Applying class weights:", clas_weight)
        clas_weight = torch.tensor([clas_weight]).squeeze().float().to(device)

    criterion = nn.CrossEntropyLoss(weight=clas_weight)
    optimiser = optim.Adam(model.parameters(), lr=hyperparameters.lr)

    all_train_loss = []
    all_train_acc = []
    all_val_loss = []
    all_val_acc = []
    best_val_acc = -np.inf
    best_train_acc = -np.inf
    overrun_counter = 0

    for e in range(hyperparameters.epochs):
        train_loss = 0.0
        model.train()

        all_y = []
        all_y_pred = []
        for batch_i, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):

            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            y = inputs[-1].to(device).detach()
            if len(x) == 1:
                x = x[0]
            optimiser.zero_grad()
            
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())
            del x
            del y

        all_train_loss.append(train_loss / len(train_loader))

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred) 
        tmp_report = classification_report(all_y.argmax(axis=1), all_y_pred.argmax(axis=1), output_dict=True)
        print(confusion_matrix(all_y.argmax(axis=1), all_y_pred.argmax(axis=1)))
        print(tmp_report)
        train_acc = tmp_report['macro avg']['precision']
        all_train_acc.append(train_acc)
        

        if x_val is not None:
            val_loss, val_acc = test_model(model, val_loader, criterion, device=device)
            all_val_loss.append(val_loss)
            all_val_acc.append(val_acc)

            acc_metric = val_acc
            best_acc_metric = best_val_acc
        else:
            acc_metric = train_acc
            best_acc_metric = best_train_acc
        if acc_metric > best_acc_metric:
            checkpoint_name = f"model_{model_name}.pt"
            # 使用torch.jit.script保存完整的模型
            scripted_model = torch.jit.script(model)
            torch.jit.save(scripted_model, os.path.join(model_dir, checkpoint_name))
            print(
                "Saving model to:",
                os.path.join(model_dir, checkpoint_name),
            )
            best_train_acc = train_acc
            if x_val is not None:
                best_val_acc = val_acc
            overrun_counter = -1

        overrun_counter += 1
        if x_val is not None:
            print(
                ": %d, Train Loss: %.8f, Train Acc: %.8f, Val Loss: %.8f, "
                "Val Acc: %.8f, overrun_counter %i"
                % (
                    e,
                    train_loss / len(train_loader),
                    train_acc,
                    val_loss,
                    val_acc,
                    overrun_counter,
                )
            )
        else:
            print(
                ": %d, Train Loss: %.8f, Train Acc: %.8f, overrun_counter %i"
                % (e, train_loss / len(train_loader), train_acc, overrun_counter)
            )
        if overrun_counter > hyperparameters.max_overrun:
            break
    return model


def test_model(model, test_loader, criterion, device=None):
    with torch.no_grad():
        if device is None:
            torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_loss = 0.0
        model.eval()

        all_y = []
        all_y_pred = []
        counter = 1
        for inputs in test_loader:
            x = inputs[:-1][0].repeat(1, 3, 1, 1)
            y = inputs[1].float()

            y_pred = model(x)

            loss = criterion(y_pred, y)

            test_loss += loss.item()

            all_y.append(y.cpu().detach())
            all_y_pred.append(y_pred.cpu().detach())

            del x
            del y
            del y_pred

            counter += 1

        all_y = torch.cat(all_y)
        all_y_pred = torch.cat(all_y_pred)

        test_loss = test_loss / len(test_loader)
        
        tmp_report = classification_report(all_y.argmax(axis=1), all_y_pred.argmax(axis=1), output_dict=True)
        print(confusion_matrix(all_y.argmax(axis=1), all_y_pred.argmax(axis=1)))
        print(tmp_report)
        test_acc = tmp_report['macro avg']['precision']
        return test_loss, test_acc
