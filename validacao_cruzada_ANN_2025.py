import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn.neural_network import MLPClassifier
import elm
from sklearn.model_selection import train_test_split

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import csv

################################################################
########       Classe ANN com ELM      #########################
################################################################
#c
class annTestELM:
    def __init__(self, dados, n_train, n_test, act_fun, num_hide, c_value, solucao):
        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino

        n_train = int(n_train)
        num_hide = int(num_hide)
        n_test = int(n_test)

        X_train = dados[dados.columns[0:dados.shape[1]-1]][0:n_train].to_numpy()
        y_train = dados[dados.columns[dados.shape[1]-1]][0:n_train].to_numpy()
        X_test = dados[dados.columns[0:dados.shape[1]-1]][n_train:n_train+n_test].to_numpy()
        y_test = dados[dados.columns[dados.shape[1]-1]][n_train:n_train+n_test].to_numpy()



        #Create and Train ANN Classifier
        model = elm.elm(hidden_units=num_hide, activation_function=act_fun, 
                        random_type='normal', x=X_train, y=y_train, C=c_value,
                        elm_type='clf')
        beta, train_accuracy, running_time = model.fit(solucao)
        #Predict the response for test dataset
        y_pred = model.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = model.score(X_test, y_test)

    def getFmeasure(self):
        return self.f_measure

################################################################
########       Script                   ########################
################################################################

#########################################
### Dados
arquivo = 'TIN_Tudo_PETR4_25894_FROM_2018_09_28_TO_2025_04_01.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)
dados.drop(['open'], axis=1, inplace=True)
dados.drop(['high'], axis=1, inplace=True)
dados.drop(['low'], axis=1, inplace=True)
dados.drop(['real_volume'], axis=1, inplace=True)
dados['direcao'] = dados['direcao'].astype('int')

##########################################
### Parametros da ANN
arquivos = ['dados1.csv', 'dados2.csv', 'dados3.csv', 'dados4.csv', 'dados6.csv']
tamanho_banch = 10
nome_arquivo = "resultado_" + str(tamanho_banch) + ".csv"

for arquivo in arquivos:
    parametros = pd.read_csv(arquivo, delimiter=';')
    parametros.drop(['Tempo'], axis=1, inplace=True)

    print(f'\n\nInicio do arquivo {arquivo}\n')

    with open(nome_arquivo, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['Act Fun', 'Num Layer', 'Nu', 'Ac Teste', 'Ac Validação', 'Ac Final'])  # Cabeçalho

    for i, af in enumerate(parametros['Act Fun']):
        match af:
            case 0:
                act_fun = 'sigmoid'
            case 1:
                act_fun = 'relu'
            case 2:
                act_fun = 'sin'
            case 3:
                act_fun = 'tanh'
            case 4:
                act_fun = 'leaky_relu'

        #Cross-Valitation for poly kernel
        f_measure_elm = []
        for idx in range(31):#len(dados)-2
            
            f = annTestELM(dados, 1240+idx*tamanho_banch, tamanho_banch, \
                        act_fun, parametros['Num Layer'][i] , parametros['Nu'][i], \
                            'no_re').getFmeasure()
            f_measure_elm.append(f)
            #print(f'Num do Banch: {idx} - F-Measure: {f}')

        f_measure_elm_mean = np.average(f_measure_elm)
        a = 1-parametros['Acuracias'][i]
        final = f_measure_elm_mean*0.8 + a*0.2

        array = parametros.values[i]

        array = array[:-1]
        array = np.append(array, a)
        array = np.append(array, f_measure_elm_mean)
        array = np.append(array, final)

        with open(nome_arquivo, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter=";")
            writer.writerow(array) 

        print(f'Teste: {a:.4f} - Validação: {f_measure_elm_mean:.4f} - Final: {final:.4f}')


#plt.plot(f_measure_elm, linewidth=0.5, color='b', label='ELM')

#plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
#plt.ylabel('F-Measure')
#plt.title('F-Measure evolution on Cross-Valitation - ELM')

#plt.savefig("test_ann_pthon_parametros.png")
#plt.show()