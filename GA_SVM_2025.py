import numpy as np
from geneticalgorithm import geneticalgorithm as ga

import pandas as pd

from algoritimos import *

import time
import multiprocessing

#####################################################
#Inicializa os dados

arquivo = 'TIN_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

dados = pd.read_csv(arquivo)
dados.drop(['Unnamed: 0'], axis=1, inplace=True)
dados.drop(['time'], axis=1, inplace=True)
dados.drop(['open'], axis=1, inplace=True)
dados.drop(['high'], axis=1, inplace=True)
dados.drop(['low'], axis=1, inplace=True)
dados.drop(['real_volume'], axis=1, inplace=True)
dados['direcao'] = dados['direcao'].astype('int')

####################################################
def minha_funcao(X, return_dict=None):
       #__init__(self, dados, n_train, act_fun, num_hide, c_value, solucao)
    #print(X)

    #deg_max = 5
    #gamma_max = 10
    #deg = int((deg_max-1)*X[2]/10)+1
    #gamma = ((gamma_max-0.1)*X[2]/10)+0.1
    match X[0]:
        case 0:
            act_fun = 'linear'
        case 1:
            act_fun = 'poly'
        case 2:
            act_fun = 'rbf'
        case 3:
            act_fun = 'sigmoid'

    
    
    return_dict['model'] = svmTest(dados, 600, act_fun, X[1], X[2], 'scale').getFmeasure()
#X[0] = kernel_type -> {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} - int
#X[1] = nu_value -> {0.1 a 0.9} - real
#X[2] = degree_value -> {0 a 10} - int
#X[2] = gamma_value -> {0.1 a 10} - real




def f(X):
    inicio = time.time()
    

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processo = multiprocessing.Process(target=minha_funcao, args=(X, return_dict))

    processo.start()  # Inicia o processo
    inicio_p = time.time()
    # Aguarda o processo terminar com um timeout
    processo.join(10)
    fim_p = time.time()

    if processo.is_alive():
        # Caso o processo ainda esteja em execução após o timeout
        processo.terminate()  # Interrompe o processo
        processo.join()  # Espera o processo terminar após ser interrompido
        #print(f"O treinamento ultrapassou o tempo limite de 10 segundos!")
        resultado = 1  # Retorna None, indicando que o processo foi interrompido
    else:
        # Caso contrário, obtém o resultado
        if 'model' in return_dict:
            resultado = 1 - return_dict['model']
        else:
            #print("Nenhum resultado foi retornado!")
            resultado = 1  # Caso o resultado não tenha sido obtido corretamente

    
    fim = time.time()
    print(f'Kernel: {X[0]} // NU: {X[1]} // Deg: {X[2]}   //  Acuracia: {resultado}   //  Tempo: {fim-inicio} //  Tempo Processo {fim_p-inicio_p}')
    #print(f'Kernel: {X[0]} // NU: {X[1]} // Deg/Gamma: {X[2]} ')
    return resultado


 
    
if __name__ == '__main__':  # Importante para evitar o erro no Windows

    inicio = time.time()

    varbound=np.array([[0,3],[0.1,0.9],[1, 4]])
    vartype=np.array([['int'],['real'],['int']])

    algorithm_param = {'max_num_iteration': 5000,\
                    'population_size':100,\
                    'mutation_probability':0.8,\
                    'elit_ratio': 0.03,\
                    'crossover_probability': 0.90,\
                    'parents_portion': 0.1,\
                    'crossover_type':'uniform',\
                    'max_iteration_without_improv':500}

    model=ga(function=f,\
            function_timeout = 20,\
            dimension=len(vartype),\
            variable_type_mixed=vartype,\
            variable_boundaries=varbound, \
            algorithm_parameters=algorithm_param)

    model.run()

    fim = time.time()

    print(f'Tempo: {fim-inicio}')

    print(model.report)
    print(model.output_dict)