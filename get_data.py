import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime



class getHistoryData:
    def __init__(self, ativo, num_dias_intervalo) -> None:
        
        mt5.initialize()


        dados = pd.DataFrame(mt5.copy_rates_from_pos(ativo, mt5.TIMEFRAME_D1, 0, num_dias_intervalo))
        dados['time']=pd.to_datetime(dados['time'], unit='s')

        dados.head()

        print(dados)

        #dados.drop(['Unnamed: 0'], axis=1, inplace=True)
        #dados.drop(["open"], axis=1, inplace=True)
        #dados.drop(["high"], axis=1, inplace=True)
        #dados.drop(["low"], axis=1, inplace=True)
        dados.drop(["tick_volume"], axis=1, inplace=True)
        dados.drop(["spread"], axis=1, inplace=True)
        #dados.drop(["real_volume"], axis=1, inplace=True)

        #dados['time'] = pd.to_datetime(dados['time'])
        #dados.index = dados['time']
        #dados.drop(["time"], axis=1, inplace=True)

        
        mt5.shutdown()

        dados.to_csv('Tudo_' + ativo + '_' + str(num_dias_intervalo) + '_FROM_' + dados['time'][0].strftime("%Y_%m_%d") + '_TO_' + dados['time'][len(dados['time'])-1].strftime("%Y_%m_%d") + '.csv')

            
#Ajustar os dias de inicio e final também!
getHistoryData('PETR4', 2520)