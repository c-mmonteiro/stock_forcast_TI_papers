import pandas as pd
import matplotlib.pyplot as plt
#Indicadores técnicos dos artigos Kara (2011), Patel (2015) e Henrique (2023)

class indicadoresTecnicos:
    def __init__(self, dados) -> None:

        self.dados = dados
        self.simpleMovingAverage(10)
        self.weightedMovingAverage(10)
        self.calcMomentum(10)
        self.stochasticAndWilliams(10, 10)
        self.calcRSI(10)
        self.calcMACD(10, 12, 26)
        self.adOscillator()
        self.calcCCI(10)
        self.direcao()

    def getSimpleMovingAverage(self):
        return self.sma   
    def getWeightedMovingAverage(self):
        return self.wma
    def getMomentum(self):
        return self.momentum
    def getStochasticK(self):
        return self.stochastic_k
    def getStochasticD(self):
        return self.stochastic_d
    def getWilliams(self):
        return self.williams
    def getRSI(self):
        return self.rsi
    def getMACD(self):
        return self.macd
    def getADOscillator(self):
        return self.ad_oscillator
    def getCCI(self):
        return self.cci
    def getDirecao(self):
        return self.direcao

    def simpleMovingAverage(self, size_window):
        soma = 0
        self.sma = []
        for idx, price in enumerate(self.dados['close']):
            if (idx >= size_window - 1):
                soma = soma + price
                self.sma.append(soma/size_window)
                soma = soma - self.dados['close'][idx + 1 - size_window]
            else:
                soma = soma + price
                self.sma.append(float("nan"))

    def weightedMovingAverage(self, size_window):
        #Calcula o denominador
        divisor = 0
        for div in range(size_window+1):
            divisor = divisor + div
        #Calcula o Indice
        valores = []
        indice = 0
        self.wma = []
        for idx, price in enumerate(self.dados['close']):
            
            #Calculo do indice
            if (idx >= size_window - 1):
                if idx == size_window-1:
                    valores.append(price)
                else:
                    valores[indice] = price
                soma_ponderada = 0
                for i in range(size_window):
                    idx_valor = indice - i
                    if idx_valor < 0:
                        idx_valor = size_window + idx_valor 
                    valor = valores[idx_valor]
                    soma_ponderada = soma_ponderada + (size_window - i)*valor

                self.wma.append(soma_ponderada/divisor)
            else:
                valores.append(price)
                self.wma.append(float("nan"))
            
            #posição do valor de fechamento mais novo
            #Trabalhar com buffer circular
            indice = indice + 1
            if (indice == size_window):
                indice = 0

    def calcMomentum(self, size_window):
        self.momentum = []
        for idx, price in enumerate(self.dados['close']):
            if idx >= size_window - 1:
                self.momentum.append(self.dados['close'][idx] - self.dados['close'][idx - size_window + 1])
            else:
                self.momentum.append(float("nan"))

    def stochasticAndWilliams(self, size_window_k, size_window_d):
        self.stochastic_k = []
        self.williams = []
        for idx, price in enumerate(self.dados['close']):
            if idx >= size_window_k - 1:
                lowest_low = min(self.dados['low'][idx - size_window_k + 1:idx])
                highest_high = max(self.dados['high'][idx - size_window_k + 1:idx])
                self.stochastic_k.append(100*((price-lowest_low)/(highest_high-lowest_low)))

                self.williams.append(100*((highest_high-price)/(highest_high-lowest_low)))
            else:
                self.stochastic_k.append(float("nan"))
                self.williams.append(float("nan"))

        self.stochastic_d = []
        soma = 0
        for idx, valor in enumerate(self.stochastic_k):
            if (idx >= size_window_k - 1):
                if (idx >= size_window_k + size_window_d - 2):
                    soma = soma + valor
                    self.stochastic_d.append(soma/size_window_d)
                    soma = soma - self.stochastic_k[idx + 1 - size_window_d]
                else:
                    soma = soma + valor
                    self.stochastic_d.append(float("nan"))
            else:
                self.stochastic_d.append(float("nan"))

    def calcRSI(self, size_window):
        soma_up = 0
        soma_down = 0
        self.rsi = []
        for idx, price in enumerate(self.dados['close']):
            if (idx > size_window - 1):
                for i in range(size_window):
                    price_change = self.dados['close'][idx-i] - self.dados['close'][idx-i-1]
                    if (price_change > 0): #UP
                        soma_up = soma_up + price_change
                    else:
                        soma_down = soma_down - price_change
                
                self.rsi.append(100 - (100/(1+(soma_up/soma_down))))
            else:
                self.rsi.append(float("nan"))

    def calcMACD(self, size_macd_window, size_short_window, size_long_window):
        k = self.dados['close'].ewm(span=size_short_window, adjust=False, min_periods=size_short_window).mean()
        d = self.dados['close'].ewm(span=size_long_window, adjust=False, min_periods=size_long_window).mean()
        macd = k - d
        macd_s = macd.ewm(span=size_macd_window, adjust=False, min_periods=size_macd_window).mean()

        self.macd = macd_s.values.tolist()


    #Formula dos artigos, diferente de outras bibliografias
    def adOscillator(self):
        self.ad_oscillator = []
        for idx, price in enumerate(self.dados['close']):
            if idx > 0:
                ado = (self.dados['high'][idx] - self.dados['close'][idx-1])/(self.dados['high'][idx] - self.dados['low'][idx])
                self.ad_oscillator.append(ado)
            else:
                self.ad_oscillator.append(float("nan"))

    def calcCCI(self, size_window):
        self.cci = []
        soma = 0
        indice = 0
        valores_mt = []
        for idx, price in enumerate(self.dados['close']):
            mt = (self.dados['high'][idx] + self.dados['low'][idx] + self.dados['close'][idx])/3
            
            #Calculo do indice
            if (idx >= size_window - 1):    
                if (idx == size_window - 1):
                    valores_mt.append(mt)
                    soma = soma + mt
                else:
                    valores_mt[indice] = mt
                    if (indice == size_window - 1):
                        soma = soma + mt - valores_mt[0]
                    else:
                        soma = soma + mt - valores_mt[indice + 1]
                
                smt = soma/size_window

                dt = 0
                for i in range(size_window):
                    idx_valor = indice - i
                    if idx_valor < 0:
                        idx_valor = size_window + idx_valor
                    dt = dt + abs(valores_mt[idx_valor] - smt)
                dt = dt/size_window
                aux = (mt-smt)/(0.015*dt)
                self.cci.append(aux)
            else:
                valores_mt.append(mt)
                soma = soma + mt
                self.cci.append(float("nan"))

            #posição do valor de fechamento mais novo
            #Trabalhar com buffer circular
            indice = indice + 1
            if (indice == size_window):
                indice = 0

    def direcao(self):
        self.direcao = []
        for idx, price in enumerate(self.dados['close']):
            if (idx > 4) and (idx < len(self.dados['close'])-1):
                if (self.dados['open'][idx] < self.dados['close'][idx+1]):
                    self.direcao.append(int(1))
                else:
                    self.direcao.append(int(-1))

    

arquivo = 'Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)

#dados.drop(['time'], axis=1, inplace=True)

ti = indicadoresTecnicos(dados)

#plt.plot(dados['close'], label='Close')

#plt.plot(ti.getSimpleMovingAverage(), label='sma')
#plt.plot(ti.getWeightedMovingAverage(), label='wma')
#plt.plot(ti.getMomentum(), label='momentum')
#plt.plot(ti.getStochasticD(), label='Stochastic D')
#plt.plot(ti.getStochasticK(), label='Stochastic K')
#plt.plot(ti.getWilliams(), label='Williams')
#plt.plot(ti.getRSI(), label='rsi')
#plt.plot(ti.getMACD(), label='macd')
#plt.plot(ti.getADOscillator(), label='A/D Oscillator')
#plt.plot(ti.getCCI(), label='CCI')

#plt.legend()
#plt.show()

#print(dados.head(15))

dados['SMA'] = pd.DataFrame(ti.getSimpleMovingAverage())
dados['WMA'] = pd.DataFrame(ti.getWeightedMovingAverage())
dados['Momentum'] = pd.DataFrame(ti.getMomentum())
dados['StochasticD'] = pd.DataFrame(ti.getStochasticD())
dados['StochasticK'] = pd.DataFrame(ti.getStochasticK())
dados['Williams'] = pd.DataFrame(ti.getWilliams())
dados['RSI'] = pd.DataFrame(ti.getRSI())
dados['MACD'] = pd.DataFrame(ti.getMACD())
dados['ADO'] = pd.DataFrame(ti.getADOscillator())
dados['CCI'] = pd.DataFrame(ti.getCCI())
dados['direcao'] = pd.DataFrame(ti.getDirecao())


print(dados.head(50))

dados = dados.dropna(axis=0)

print(dados)

dados.to_csv('TI_' + arquivo)




