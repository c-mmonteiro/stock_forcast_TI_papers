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
    


################################################################
########       Ensable SVM (Poly e rbf) + ANN   ################
################################################################
class ensableTest:
    def __init__(self, dados, n_train, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann) -> None:

        #Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train)
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train)
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test)
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test)
        y_test = y_test.values.tolist()

        #Create Models
        clf_poly = svm.NuSVC(kernel='poly', nu=nu_poly, degree=degree_poly, gamma=gamma_poly) 
        clf_rbf = svm.NuSVC(kernel='rbf', nu=nu_rbf, degree=3, gamma=gamma_rbf) 
        model = elm.elm(hidden_units=num_hide_ann, activation_function=act_fun_ann, 
                        random_type='normal', x=X_train, y=y_train, C=c_ann,
                        elm_type='clf')

        #Train the model using the training sets
        clf_poly.fit(X_train, y_train)
        clf_rbf.fit(X_train, y_train)
        beta, train_accuracy, running_time = model.fit(solucao_ann)
        #Predict the response for test dataset
        y_pred_svm_poly = clf_poly.predict(X_test)
        y_pred_svm_rbf = clf_rbf.predict(X_test)
        y_pred_ann = model.predict(X_test)

        y_pred = []
        for i in range(y_pred_ann.shape[0]):
            soma = y_pred_ann[i] + y_pred_svm_rbf[i] + y_pred_svm_poly[1]
            if soma > 1:
                y_pred.append(1)
            else:
                y_pred.append(0)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = metrics.accuracy_score(y_test, y_pred)
        self.f_measure_svm_poly = metrics.accuracy_score(y_test, y_pred_svm_poly)
        self.f_measure_svm_rbf = metrics.accuracy_score(y_test, y_pred_svm_rbf)
        self.f_measure_ann = metrics.accuracy_score(y_test, y_pred_ann)

    
    def getFmeasure(self):
        return self.f_measure
    
    def getFmeasurePoly(self):
        return self.f_measure_svm_poly
    
    def getFmeasureRbf(self):
        return self.f_measure_svm_rbf
    
    def getFmeasureAnn(self):
        return self.f_measure_ann
    

################################################################
########       Ensable SVM (Poly e rbf) + ANN   ################
########                Previsão D + 1          ################
################################################################
class testD1:
    def __init__(self, dados, n_train, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann) -> None:

        #Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train)
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train)
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test)
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test)
        y_test = y_test.values.tolist()

        #Create Models
        clf_poly = svm.NuSVC(kernel='poly', nu=nu_poly, degree=degree_poly, gamma=gamma_poly) 
        clf_rbf = svm.NuSVC(kernel='rbf', nu=nu_rbf, degree=3, gamma=gamma_rbf) 
        model = elm.elm(hidden_units=num_hide_ann, activation_function=act_fun_ann, 
                        random_type='normal', x=X_train, y=y_train, C=c_ann,
                        elm_type='clf')

        #Train the model using the training sets
        clf_poly.fit(X_train, y_train)
        clf_rbf.fit(X_train, y_train)
        beta, train_accuracy, running_time = model.fit(solucao_ann)
        #Predict the response for test dataset
        y_pred_svm_poly = clf_poly.predict(X_test)
        y_pred_svm_rbf = clf_rbf.predict(X_test)
        y_pred_ann = model.predict(X_test)

        soma = y_pred_ann[0] + y_pred_svm_rbf[0] + y_pred_svm_poly[0]
        if soma > 1:
            y_pred_ensamble = 1
        else:
            y_pred_ensamble = 0
        
        if soma == 3:
            y_pred_ensamble_unanimidade = 1
        elif soma == 0:
            y_pred_ensamble_unanimidade = 0
        else:
            y_pred_ensamble_unanimidade = -1

        self.y_pred_acerto = []
        if y_test[0] == y_pred_ensamble:
            self.y_pred_acerto.append(1)
        else:
            self.y_pred_acerto.append(0)

        if y_pred_ensamble_unanimidade == -1:
            self.y_pred_acerto.append(-1)
        else:
            if y_test[0] == y_pred_ensamble_unanimidade:
                self.y_pred_acerto.append(1)
            else:
                self.y_pred_acerto.append(0)

        if y_test[0] == y_pred_svm_poly[0]:
            self.y_pred_acerto.append(1)
        else:
            self.y_pred_acerto.append(0)

        if y_test[0] == y_pred_svm_rbf[0]:
            self.y_pred_acerto.append(1)
        else:
            self.y_pred_acerto.append(0)

        if y_test[0] == y_pred_ann[0]:
            self.y_pred_acerto.append(1)
        else:
            self.y_pred_acerto.append(0)

    
    def getAcertos(self):
        return self.y_pred_acerto


    
    
    
