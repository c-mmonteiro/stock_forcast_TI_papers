import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#ANN Library
import elm

#XGBoost
from xgboost import XGBClassifier

################################################################
########       Classe SVM          #############################
################################################################
class svmTest:
    def __init__(self, dados, n_train, kernel_type, nu_value, degree_value, gamma_value) -> None:

        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train)
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train)
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test)
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test)
        y_test = y_test.values.tolist()

        #Create a svm Classifier
        clf = svm.NuSVC(kernel=kernel_type, nu=nu_value, degree=degree_value, gamma=gamma_value) 
        #Train the model using the training sets
        clf.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = metrics.accuracy_score(y_test, y_pred)
    
    def getFmeasure(self):
        return self.f_measure
    
################################################################
########       Classe ANN com ELM      #########################
################################################################
#https://github.com/5663015/elm
class annTestELM:
    def __init__(self, dados, n_train, act_fun, num_hide, c_value, solucao):
        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train).to_numpy()
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train).to_numpy()
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test).to_numpy()
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test).to_numpy()


        #Create and Train ANN Classifier
        model = elm.elm(hidden_units=num_hide, activation_function=act_fun, 
                        random_type='normal', x=X_train, y=y_train, C=c_value,
                        elm_type='clf')
        beta, train_accuracy, running_time = model.fit(solucao)
        #Predict the response for test dataset
        #y_pred = model.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = model.score(X_test, y_test)

    def getFmeasure(self):
        return self.f_measure
    
################################################################
########       Classe XGBoost          #########################
################################################################
#https://dadosaocubo.com/o-guia-do-xgboost-com-python/
class xgBoostTest:
    #FAZER TUDO
    def __init__(self, dados, n_train, booster_type, eta_value, colsample_value, 
                 lambda_value, alpha_value, estimators_value, depth_value, 
                 subsample_value):
        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train).to_numpy()
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train).to_numpy()
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test).to_numpy()
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test).to_numpy()


        #Create and Train XGBoost Classifier
        clf = XGBClassifier(booster=booster_type,
                            eta = eta_value,
                            colsample_bytree= colsample_value,
                            reg_lambda = lambda_value,
                            alpha = alpha_value,
                            n_estimators = estimators_value,
                            max_depth = depth_value, 
                            eval_metric = 'error',
                            subsample = subsample_value,
                            objective = 'binary:logistic', 
                            random_state=0)
        

        clf.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = metrics.accuracy_score(y_test, y_pred)

    def getFmeasure(self):
        return self.f_measure