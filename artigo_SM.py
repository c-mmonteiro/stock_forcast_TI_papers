import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

class svmTest:
    def __init__(self, dados, n_train, kernel_type) -> None:

        # Split dataset into training set and test set
        #n_train => Numero de amostras no treino
        n_test = len(dados) - n_train

        X_train = dados[dados.columns[0:dados.shape[1]-1]].head(n_train)
        y_train = dados[dados.columns[dados.shape[1]-1]].head(n_train)
        X_test = dados[dados.columns[0:dados.shape[1]-1]].tail(n_test)
        y_test = dados[dados.columns[dados.shape[1]-1]].tail(n_test)
        y_test = y_test.values.tolist()

        #Create a svm Classifier
        clf = svm.NuSVC(kernel=kernel_type, decision_function_shape='ovo') # Linear Kernel
        #Train the model using the training sets
        clf.fit(X_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(X_test)

        # Model Accuracy: how often is the classifier correct?
        self.f_measure = metrics.accuracy_score(y_test, y_pred)
    
    def getFmeasure(self):
        return self.f_measure
    
    

arquivo = 'TI_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)
dados.drop(['open'], axis=1, inplace=True)
dados.drop(['high'], axis=1, inplace=True)
dados.drop(['low'], axis=1, inplace=True)
dados.drop(['real_volume'], axis=1, inplace=True)
dados['direcao'] = dados['direcao'].astype('int')

#Cross-Valitation for poly kernel
f_measure_poly = []
for idx in range(600, 840):#len(dados)-2
    f = svmTest(dados, idx, 'poly').getFmeasure()
    f_measure_poly.append(f)
    print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

f_measure_poly_mean = np.average(f_measure_poly)
print(f'F-Measure médio: {f_measure_poly_mean}')

#Cross-Valitation for radial kernel
f_measure_radial = []
for idx in range(600, 840):
    f = svmTest(dados, idx, 'rbf').getFmeasure()
    f_measure_radial.append(f)
    print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

f_measure_radial_mean = np.average(f_measure_radial)
print(f'F-Measure médio RBF: {f_measure_radial_mean}')
print(f'F-Measure médio Poly: {f_measure_poly_mean}')



plt.plot(f_measure_poly, linewidth=0.5, color='b', label='Poly')
plt.plot(f_measure_radial, linewidth=0.5, color='r', label='Radial')
plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
plt.ylabel('F-Measure')
plt.title('F-Measure evolution on Cross-Valitation')
plt.legend(title='Kernel:')
plt.savefig("test_svm_pthon.png")
plt.show()
