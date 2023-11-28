import pandas as pd
import matplotlib.pyplot as plt


arquivo = 'parametros_ELM.csv'

dados = pd.read_csv(arquivo)

dados = dados.sort_values("Fmeasure", ascending=False)

print(dados)

#print(dados[dados["Kernel"] == "poly"].head())
#print(dados[dados["Kernel"] == "rbf"].head())


#maior = dados['Fmeasure'].max()
#menor = dados['Fmeasure'].min()
#idx = dados['Fmeasure'].idxmax()

#print(f'Maior: {maior}  \nMenor: {menor}')
#print(dados.iloc[[dados['Fmeasure'].idxmax()]])
#print(dados.iloc[[dados['Fmeasure'].idxmin()]])
