import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from datetime import datetime


class leituraDados:
    def __init__(self, file) -> None:
        
        self.dados = pd.read_csv(file)

        self.dados.head()

        self.dados.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.dados.drop(["open"], axis=1, inplace=True)
        self.dados.drop(["high"], axis=1, inplace=True)
        self.dados.drop(["low"], axis=1, inplace=True)
        self.dados.drop(["tick_volume"], axis=1, inplace=True)
        self.dados.drop(["spread"], axis=1, inplace=True)
        #self.dados.drop(["real_volume"], axis=1, inplace=True)

        self.dados['time'] = pd.to_datetime(self.dados['time'])
        self.dados.index = self.dados['time']
        self.dados.drop(["time"], axis=1, inplace=True)


    def getDados(self):  

        return self.dados
    
class tratamentoDados:
    def __init__(self, dados) -> None:

        self.dados = dados


        self.dados['Return'] = self.dados['close'].pct_change()
        self.dados['LogRet'] = np.log(self.dados['close']).diff()
        self.MA20_dados = self.dados.close.rolling(window=20).mean()
        self.MA100_dados = self.dados.close.rolling(window=100).mean()

        self.dados = self.dados.dropna()
        self.MA20_dados = self.MA20_dados.dropna()
        self.MA100_dados = self.MA100_dados.dropna()

        self.dados_W = self.dados.resample(rule='W').last()
        self.dados_M = self.dados.resample(rule='M').last()
        
        
        fig, ax = plt.subplots(3,1, figsize=(12,8))
        fig.autofmt_xdate()

        ax[0].set_title("Preço de Fechamento")
        ax[0].plot(self.dados.index, self.dados.close, label='Ação', color='tab:blue')
        ax[0].plot(self.MA20_dados.index, self.MA20_dados, label='MA20', color='tab:red')
        ax[0].plot(self.MA100_dados.index, self.MA100_dados, label='MA100', color='tab:green')
        ax[0].legend()

        ax[1].set_title("Retorno")
        ax[1].plot(self.dados.index, self.dados.Return, label='Ação', color='tab:blue')

        ax[2].set_title("Retorno Logaritmico")
        ax[2].plot(self.dados.index, self.dados.LogRet, label='Ação', color='tab:blue')


        print("Auto correlação fechamento Diário: " + str(self.dados['close'].autocorr()))
        print("Auto correlação fechamento Semanal: " + str(self.dados_W['close'].autocorr()))
        print("Auto correlação fechamento Mensal: " + str(self.dados_M['close'].autocorr()))
        print("---------------------------------")
        print("Auto correlação retorno Diário: " + str(self.dados['Return'].autocorr()))
        print("Auto correlação retorno Semanal: " + str(self.dados_W['Return'].autocorr()))
        print("Auto correlação retorno Mensal: " + str(self.dados_M['Return'].autocorr()))

        fig2, ax2 = plt.subplots(3,2, figsize=(15,8))
        plot_acf(self.dados['close'], lags=498, alpha=0.05, ax = ax2[0, 0])
        plot_acf(self.dados['Return'], lags=498, alpha=0.05, ax = ax2[0, 1])
        plot_acf(self.dados_W['close'], lags=104, alpha=0.05, ax = ax2[1, 0])
        plot_acf(self.dados_W['Return'], lags=104, alpha=0.05, ax = ax2[1, 1])
        plot_acf(self.dados_M['close'], lags=23, alpha=0.05, ax = ax2[2, 0])
        plot_acf(self.dados_M['Return'], lags=23, alpha=0.05, ax = ax2[2, 1])
        #plt.show()

    def getDiario(self):
        return self.dados
    
    def getSemanal(self):
        return self.dados_W
    
    def getMensal(self):
        return self.dados_M


class modeloARMA:
    def __init__(self, dados) -> None:
        
        ar=np.array([1,0.32])
        ma=np.array([1])
        AR_object =  ArmaProcess(ar,ma)
        simulated_data=AR_object.generate_sample(nsample=1000)

        mod = ARMA(dados['Return'], order=(1,0))
        result=mod.fit()
        result.summary()

        fig, ax = plt.subplots(figsize=(12,6))
        g = result.plot_predict(start='2023-08-01', end='2023-09-12',ax=ax)

class modeloARIMA:
    def __init__(self, dados) -> None:
        
        mod = ARIMA(dados['close'].values, order=(2,1,1))
        res = mod.fit()
        print(res.summary())
        pre = res.predict(start=0, end=520)
        print(pre)
        print(dados['close'])

        fig, ax = plt.subplots()
        ax.plot(dados['close'].values.tolist())
        
        ax.plot(pre)
        plt.show()


        #fig, ax = plt.subplots(figsize=(12,6))
        #g  = res.plot_predict(start='2023-08-01', end='2023-09-12',ax=ax)
        


dados = leituraDados('PETR4_500_FROM_2021_08_31_TO_2023_08_31.csv').getDados()
dados = tratamentoDados(dados).getDiario()
print(dados)

modeloARIMA(dados)

