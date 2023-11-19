import pandas as pd
import matplotlib.pyplot as plt
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

arquivo = 'SVM_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)

#print(dados[dados.columns[21:22]].head(20))


# Split dataset into training set and test set
n_train = 600     #Numero de amostras no treino
n_test = len(dados) - n_train

X_train = dados[dados.columns[0:21]].head(n_train)
y_train = dados[dados.columns[21:22]].head(n_train)
X_test = dados[dados.columns[0:21]].tail(n_test)
y_test = dados[dados.columns[21:22]].tail(n_test)

plt.plot(dados['open'], linewidth=3, color='k')

plt.plot(X_train['open'], linewidth=0.5, color='b')

plt.plot(X_test['open'], linewidth=0.5, color='r')

plt.show()


#Create a svm Classifier
clf = svm.NuSVC(kernel='poly') # Linear Kernel
#Train the model using the training sets
clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)

y_test = y_test.values.tolist()

baixa_certo = 0
baixa_errado = 0
alta_certo = 0
alt_errado = 0
for idx, y in enumerate(y_pred):
    if ((y == -1) and (y_test[idx][0] == -1)):
        baixa_certo = baixa_certo + 1
    elif ((y == -1) and (y_test[idx][0] == 1)):
        baixa_errado = baixa_errado + 1
    elif ((y == 1) and (y_test[idx][0] == 1)):
        alta_certo = alta_certo + 1
    elif ((y == 1) and (y_test[idx][0] == -1)):
        alt_errado = alt_errado + 1

print(f'Baixa certo: {baixa_certo}  //  Baixa Errado: {baixa_errado}')
print(f'Alta Errado: {alt_errado}   //  Alta Certo: {alta_certo}')

total_certo = ((alta_certo + baixa_certo)/n_test)
print(f'Acertos Total: {total_certo}    //  Alta Certo: {alta_certo/(alta_certo+alt_errado)}     //  Baixa Certo: {baixa_certo/(baixa_certo+baixa_errado)}')


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
