import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class leituraDados:
    def __init__(self, file) -> None:
        
        self.dados = pd.read_csv(file)

        self.dados.head()

        self.dados.drop(['Unnamed: 0'], axis=1, inplace=True)

        self.dados['time'] = pd.to_datetime(self.dados['time'])
        self.dados.index = self.dados['time']
        self.dados.drop(["time"], axis=1, inplace=True)


    def getDados(self):  

        return self.dados
    

dados = leituraDados('Preco_PETR4_500_FROM_2021_09_01_TO_2023_09_01.csv').getDados()
lista =[]
medias =[]
for i1 in range(22):
    soma = 0
    coluna = []
    for i2 in range(22):
        coluna.append(dados[i2]['close'])
        soma = soma + dados[i1*22 + i2]['close']
    lista.append(coluna)
    medias.append(soma/22)



print(dados)