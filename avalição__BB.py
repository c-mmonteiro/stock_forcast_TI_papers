import pandas as pd
import matplotlib.pyplot as plt


class testeBB:
    def __init__(self, dados, window_size) -> None:
        self.dados = dados

        size_dados = len(self.dados['close'])

        self.dados['MA'] = self.dados['close'].rolling(window=window_size).mean()
        self.dados['STD'] = self.dados['close'].rolling(window=window_size).std()
        
        soma_fora_1dia = 0
        soma_fora_2dia = 0
        soma_fora_3dia = 0
        soma_fora_4dia = 0
        soma_fora_5dia = 0
        for idx, price in enumerate(self.dados['close']):
            if (idx >= window_size) and (idx < size_dados - 4):
                max = self.dados['MA'][idx-1] + 2*self.dados['STD'][idx-1]
                min = self.dados['MA'][idx-1] - 2*self.dados['STD'][idx-1]
                if ((price > max) or (price < min)):
                    soma_fora_1dia = soma_fora_1dia + 1
                    
                    max2 = self.dados['MA'][idx] + 2*self.dados['STD'][idx]
                    min2 = self.dados['MA'][idx] - 2*self.dados['STD'][idx]
                    if ((self.dados['close'][idx+1] > max2) or (self.dados['close'][idx+1] < min2)):
                        soma_fora_2dia = soma_fora_2dia + 1

                        max3 = self.dados['MA'][idx + 1] + 2*self.dados['STD'][idx + 1]
                        min3 = self.dados['MA'][idx + 1] - 2*self.dados['STD'][idx + 1]
                        if ((self.dados['close'][idx + 2] > max3) or (self.dados['close'][idx + 2] < min3)):
                            soma_fora_3dia = soma_fora_3dia + 1

                            max4 = self.dados['MA'][idx + 2] + 2*self.dados['STD'][idx + 2]
                            min4 = self.dados['MA'][idx + 2] - 2*self.dados['STD'][idx + 2]
                            if ((self.dados['close'][idx + 3] > max4) or (self.dados['close'][idx + 3] < min4)):
                                soma_fora_4dia = soma_fora_4dia + 1

                                max5 = self.dados['MA'][idx + 3] + 2*self.dados['STD'][idx + 3]
                                min5 = self.dados['MA'][idx + 3] - 2*self.dados['STD'][idx + 3]
                                if ((self.dados['close'][idx + 4] > max5) or (self.dados['close'][idx + 4] < min5)):
                                    soma_fora_5dia = soma_fora_5dia + 1
        
        self.erro_1dia = 100*soma_fora_1dia/(size_dados - window_size - 4)
        self.erro_2dia = 100*soma_fora_2dia/(size_dados - window_size - 4)
        self.erro_3dia = 100*soma_fora_3dia/(size_dados - window_size - 4)
        self.erro_4dia = 100*soma_fora_4dia/(size_dados - window_size - 4)
        self.erro_5dia = 100*soma_fora_5dia/(size_dados - window_size - 4)

    def getErro(self):
        return self.erro_1dia
    
    def getErro2(self):
        return self.erro_2dia

    def getErro3(self):
        return self.erro_3dia
    
    def getErro4(self):
        return self.erro_4dia
    
    def getErro5(self):
        return self.erro_5dia
        

arquivo = 'Preco_PETR4_1240_FROM_2018_09_10_TO_2023_09_08.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)

listErro1 = []
listErro2 = []
listErro3 = []
listErro4 = []
listErro5 = []
listErroMedio = []
for i in range(2, 150):
    e = testeBB(dados, i).getErro()
    e2 = testeBB(dados, i).getErro2()
    e3 = testeBB(dados, i).getErro3()
    e4 = testeBB(dados, i).getErro4()
    e5 = testeBB(dados, i).getErro5()
    em = (e + e2 + e3 + e4 + e5)/5
    listErro1.append(e)
    listErro2.append(e2)
    listErro3.append(e3)
    listErro4.append(e4)
    listErro5.append(e5)
    listErroMedio.append(em)
    print(f'Tamanho: {i}    Erro: {e}')

plt.plot(listErro1, linewidth=0.5, color='c', label='1 dia')
plt.plot(listErro2, linewidth=0.5, color='b', label='2 dia')
plt.plot(listErro3, linewidth=0.5, color='k', label='3 dia')
plt.plot(listErro4, linewidth=0.5, color='g', label='4 dia')
plt.plot(listErro5, linewidth=0.5, color='y', label='5 dia')
plt.plot(listErroMedio, linewidth=0.5, color='r', label='Media')
plt.xlabel('Tamanho MA')
plt.ylabel('Erro (%)')
plt.title('Quantidade de fechamento fora da Banda de Bollinger')
plt.legend(loc='upper right')

plt.show()
