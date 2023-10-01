import pandas as pd

arquivo = 'Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)

df3 = pd.DataFrame(columns=['open',
                            'open1', 'high1', 'low1', 'close1',  'volume1',
                            'open2', 'high2', 'low2', 'close2', 'volume2',
                            'open3', 'high3', 'low3', 'close3', 'volume3',
                            'open4', 'high4', 'low4', 'close4', 'volume4'])

direcao = []
for idx, price in enumerate(dados['close']):
    if (idx > 4) and (idx < len(dados['close'])-1):
        if (dados['open'][idx] < dados['close'][idx+1]):
            direcao.append(1)
        else:
            direcao.append(-1)

        df3 = df3.append(
                {"open": dados['open'][idx], 
                 "open1": dados['open'][idx-1], "high1": dados['high'][idx-1], "low1": dados['low'][idx-1], "close1": dados['close'][idx-1], "volume1": dados['real_volume'][idx-1],
                 "open2": dados['open'][idx-2], "high2": dados['high'][idx-2], "low2": dados['low'][idx-2], "close2": dados['close'][idx-2], "volume2": dados['real_volume'][idx-2],
                 "open3": dados['open'][idx-3], "high3": dados['high'][idx-3], "low3": dados['low'][idx-3], "close3": dados['close'][idx-3], "volume3": dados['real_volume'][idx-3],
                 "open4": dados['open'][idx-4], "high4": dados['high'][idx-4], "low4": dados['low'][idx-4], "close4": dados['close'][idx-4], "volume4": dados['real_volume'][idx-4],
                }, ignore_index=True)

df3['direcao'] = direcao
print(df3.head())
print(df3.tail())

df3.to_csv("SVM_" + arquivo)
