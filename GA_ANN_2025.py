import numpy as np
from geneticalgorithm import geneticalgorithm as ga

import pandas as pd

from algoritimos import *

import time
import csv

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

def f(X):
    #__init__(self, dados, n_train, act_fun, num_hide, c_value, solucao)
    match X[0]:
        case 0:
            act_fun = 'sigmoid'
        case 1:
            act_fun = 'relu'
        case 2:
            act_fun = 'sin'
        case 3:
            act_fun = 'tanh'
        case 4:
            act_fun = 'leaky_relu'


    return 1-annTestELM(dados, 600, act_fun, X[1], X[2], 'no_re').getFmeasure()
#n_train -> {250 a 1000} - int
#X[0] = act_fun -> {'sigmoid', 'relu', 'sin', 'tanh', 'leaky_relu'}
#X[1] = num_hide -> {2 a 500} - int
#X[2] = c_value -> {0.1 a 0.9} - real
#solucao -> {'no_re', 'solution1', 'solution2'}

#############################################################
#############################################################
##          Inicio do script
#############################################################
arquivo_csv = 'dados6.csv'

varbound=np.array([[0,4],[2,500],[0.1, 0.9]])
vartype=np.array([['int'],['int'],['real']])

algorithm_param = {'max_num_iteration': 5000,\
                   'population_size':150,\
                   'mutation_probability':0.5,\
                   'elit_ratio': 0.02,\
                   'crossover_probability': 0.7,\
                   'parents_portion': 0.1,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':250}

print(algorithm_param)

with open(arquivo_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(['Act Fun', 'Num Layer', 'Nu', 'Acuracias', 'Tempo'])  # Cabeçalho

for idx in range(10):

    inicio = time.time()

    model=ga(function=f,\
            function_timeout = 20,\
            dimension=len(vartype),\
            variable_type_mixed=vartype,\
            variable_boundaries=varbound, \
            algorithm_parameters=algorithm_param,\
            convergence_curve=False)

    model.run()

    fim = time.time()

    resultado = model.output_dict

    array = np.append(resultado['variable'], resultado['function'])

    array = np.append(array, (fim-inicio))
    print(array)

    with open(arquivo_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(array) 

    

###################################################################