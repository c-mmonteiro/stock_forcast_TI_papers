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

##############################################################
####        Classe com SKLearn      ##########################
##############################################################
#Fazer ajustes para aANN
class annTestSK:
    def __init__(self, dados, n_train, solver_type, alpha_value, num_hidden_layer, 
                 tolerance_value, learning_rate_value, learning_rate_type, max_inter_value) -> None:

        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train)
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train)
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test)
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test)
        y_test = y_test.values.tolist()

        #Create a ANN Classifier
        clf = MLPClassifier(solver=solver_type, alpha=alpha_value, 
                            hidden_layer_sizes=num_hidden_layer, 
                            shuffle=False,
                            tol=tolerance_value,
                            batch_size=n_train,
                            learning_rate=learning_rate_type,
                            learning_rate_init=learning_rate_value,
                            max_iter= max_inter_value)
        #Train the model using the training sets
        clf.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = metrics.accuracy_score(y_test, y_pred)
    
    def getFmeasure(self):
        return self.f_measure
    
 ################################################################
 ########       Classe com ELM      #############################
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



arquivo = 'TIN_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)
dados.drop(['open'], axis=1, inplace=True)
dados.drop(['high'], axis=1, inplace=True)
dados.drop(['low'], axis=1, inplace=True)
dados.drop(['real_volume'], axis=1, inplace=True)
dados['direcao'] = dados['direcao'].astype('int')

####################################################################
#Testa os parametro da ELM ANN
####################################################################
#f_measure_parameters = pd.DataFrame(columns=['Solucao', 'ActFun', 'NumHiden', 'C', 'Fmeasure'])

#print(f'Solucao     ActFun  NumHiden    C     F-Measure')
#for solucao_test in ['no_re', 'solution1', 'solution2']:
  #  for act_fun_test in ['sigmoid', 'relu', 'sin', 'tanh', 'leaky_relu']:
 #       for num_hide_test in [10, 20, 30, 40, 50, 100]:
 #           for c_test in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
 #               f = annTestELM(dados, 600, act_fun_test, num_hide_test, c_test, solucao_test).getFmeasure()
 #               f_measure_parameters = f_measure_parameters.append(
#                    {"Solucao": solucao_test,"ActFun": act_fun_test, 
#                     "NumHiden": num_hide_test, "C": c_test, 
#                     "Fmeasure": f}, ignore_index=True)         
#                print(f'{solucao_test}  {act_fun_test}   {num_hide_test}   {c_test}    {f}')

#f_measure_parameters.to_csv('parametros_ELM.csv')
######################################################################

#Cross-Valitation for poly kernel
f_measure_elm = []
for idx in range(600, 840):#len(dados)-2
    f = annTestELM(dados, idx, 'sigmoid', 10, 0.4, 'no_re').getFmeasure()
    f_measure_elm.append(f)
    print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

f_measure_elm_mean = np.average(f_measure_elm)
print(f'F-Measure médio: {f_measure_elm_mean}')


plt.plot(f_measure_elm, linewidth=0.5, color='b', label='ELM')
#plt.plot(f_measure_radial, linewidth=0.5, color='r', label='Radial')
plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
plt.ylabel('F-Measure')
plt.title('F-Measure evolution on Cross-Valitation - ELM')
#plt.legend(title='Kernel:')
plt.savefig("test_ann_pthon_parametros.png")
plt.show()