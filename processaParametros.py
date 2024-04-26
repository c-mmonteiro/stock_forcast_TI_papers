import pandas as pd
import matplotlib.pyplot as plt

class processaParametrosSVM:
    def __init__(self, arquivo):
        dados = pd.read_csv(arquivo)

        dados = dados.sort_values("Fmeasure", ascending=False)
        print('--------------------------------')
        print(arquivo)

        print(dados[dados["Kernel"] == "poly"].head(3))
        print(dados[dados["Kernel"] == "rbf"].head(3))

        dados_poly = dados[dados["Kernel"] == "poly"]

        maior_poly = dados_poly["Fmeasure"].max()
        menor_poly = dados_poly["Fmeasure"].min()

        dados_rbf = dados[dados["Kernel"] == "rbf"]

        maior_rbf = dados_rbf["Fmeasure"].max()
        menor_rbf = dados_rbf["Fmeasure"].min()
        print(f'F-Measure Poly Max: {maior_poly} - Min: {menor_poly}')
        print(f'F-Measure Rbf Max: {maior_rbf} - Min: {menor_rbf}')
        print('--------------------------------')

class processaParametrosANN:
    def __init__(self, arquivo):
        dados = pd.read_csv(arquivo)

        dados = dados.sort_values("Fmeasure", ascending=False)
        print('--------------------------------')
        print(arquivo)

        print(dados.head(1))

        maior = dados['Fmeasure'].max()
        menor = dados['Fmeasure'].min()
        print(f'F-Measure Max: {maior} - Min: {menor}')
        print('--------------------------------')
        


#processaParametrosANN('parametros_ANN_ITUB4.csv')
#processaParametrosANN('parametros_ANN_PETR4.csv')
#processaParametrosANN('parametros_ANN_VALE3.csv')
#processaParametrosANN('parametros_ANN_BBAS3.csv')

#processaParametrosSVM('parametros_SVM_ITUB4.csv')
#processaParametrosSVM('parametros_SVM_PETR4.csv')
#processaParametrosSVM('parametros_SVM_BBAS3.csv')
processaParametrosSVM('parametros_SVM_VALE3.csv')





