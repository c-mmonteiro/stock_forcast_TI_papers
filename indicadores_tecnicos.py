import pandas as pd


class indicadores_tecnicos:
    def __init__(self, dados) -> None:

        self.dados = dados



arquivo = 'Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)

indicadores_tecnicos(dados)

