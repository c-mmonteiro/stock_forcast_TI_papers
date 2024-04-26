import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from algoritimos import *  
from indicadores_tecnicos import *  
import csv


class avaliacao:
    def __init__(self, arquivo):     
        self.dados = pd.read_csv(arquivo)
        self.dados.drop(['Unnamed: 0'], axis=1, inplace=True)

        self.dados = indicadoresTecnicos(self.dados).updateDataFrameNormal(self.dados)

        self.dados.drop(['time'], axis=1, inplace=True)
        self.dados.drop(['open'], axis=1, inplace=True)
        self.dados.drop(['high'], axis=1, inplace=True)
        self.dados.drop(['low'], axis=1, inplace=True)
        self.dados.drop(['real_volume'], axis=1, inplace=True)
        self.dados['direcao'] = self.dados['direcao'].astype('int')

        contagem = self.dados['direcao'].value_counts()
        print('------------------------------')
        print(arquivo)
        print(f"Baixa: {contagem[0]} - {contagem[0]/len(self.dados)}")
        print(f"Alta: {contagem[1]} - {contagem[1]/len(self.dados)}")

        #Visualizar a distribuição dos dados
        #colunas = self.dados.columns.to_list()
        #plt.figure(figsize=(15,8))
        #self.dados.boxplot(column = colunas[:-1])
        #plt.show()

####################################################################
#SVM
####################################################################
    def testParametrosSVM(self, salvar, nome_arquivo):
        f_measure_parameters = pd.DataFrame(columns=['Kernel', 'Nu', 'Gamma', 'Degree', 'Fmeasure'])

        print(f'Kernel  Nu  Gamma   Degree  F-Measure')
        for kernel_test in ['poly', 'rbf']:
            for nu_test in [0.1, 0.25, 0.5, 0.75, 0.9]:
                for gamma_test in [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 10]:
                    if kernel_test == 'poly':
                        for degree_test in [1, 2, 3, 4, 5]:
                            f = svmTest(self.dados, 600, kernel_test, nu_test, degree_test, gamma_test).getFmeasure()
                            f_measure_parameters = f_measure_parameters.append(
                                {"Kernel": kernel_test, "Nu": nu_test, "Gamma": gamma_test,
                                "Degree": degree_test, "Fmeasure": f}, ignore_index=True)         
                            print(f'{kernel_test}   {nu_test}   {gamma_test}    {degree_test}   {f}')
                    else:
                        f = svmTest(self.dados, 600, kernel_test, nu_test, 3, gamma_test).getFmeasure()
                        f_measure_parameters = f_measure_parameters.append(
                                {"Kernel": kernel_test, "Nu": nu_test, "Gamma": gamma_test,
                                "Degree": 3, "Fmeasure": f}, ignore_index=True)
                        print(f'{kernel_test}   {nu_test}   {gamma_test}    NaN     {f}')
        if (salvar > 0):
            f_measure_parameters.to_csv(nome_arquivo)

        return f_measure_parameters.sort_values("Fmeasure", ascending=False)
    
    def crossValitationSVM(self, save, ativo, kernel_cv, nu_cv, degree_cv, gamma_cv):
        #Cross-Valitation for poly kernel
        print('----------------------------')
        print(f'Validação Cruzada SVM do {ativo} com kernel {kernel_cv}')
        f_measure = []
        for idx in range(600, 840):#len(dados)-2
            f = svmTest(self.dados, idx, kernel_cv, nu_cv, degree_cv, gamma_cv).getFmeasure()
            f_measure.append(f)
            print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_mean = np.average(f_measure)
        print(f'F-Measure médio {kernel_cv}: {f_measure_mean}')

        plt.figure()
        plt.plot(range(600, 840), f_measure, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        if kernel_cv == 'poly':
            plt.title('F-Measure evolution on Cross-Valitation for SVM Polynomial - ' + ativo)
        else:
            plt.title('F-Measure evolution on Cross-Valitation for SVM Radial - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('SVM_' + kernel_cv + '_' + ativo)
        #plt.show()

####################################################################
#ANN
####################################################################
    def testParametrosANN(self, salvar, nome_arquivo):
        f_measure_parameters = pd.DataFrame(columns=['Solucao', 'ActFun', 'NumHiden', 'C', 'Fmeasure'])

        print(f'Solucao     ActFun  NumHiden    C     F-Measure')
        for solucao_test in ['no_re', 'solution1', 'solution2']:
            for act_fun_test in ['sigmoid', 'relu', 'sin', 'tanh', 'leaky_relu']:
                for num_hide_test in [10, 20, 30, 40, 50, 100]:
                    for c_test in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        f = annTestELM(self.dados, 600, act_fun_test, num_hide_test, c_test, solucao_test).getFmeasure()
                        f_measure_parameters = f_measure_parameters.append(
                            {"Solucao": solucao_test,"ActFun": act_fun_test, 
                                "NumHiden": num_hide_test, "C": c_test, 
                                "Fmeasure": f}, ignore_index=True)         
                        print(f'{solucao_test}  {act_fun_test}   {num_hide_test}   {c_test}    {f}')

        if (salvar > 0):
            f_measure_parameters.to_csv(nome_arquivo)

        return f_measure_parameters.sort_values("Fmeasure", ascending=False)
    
    def crossValitationANN(self, save, ativo, solucao_cv, act_fun_cv, num_hide_cv, c_cv):
        #Cross-Valitation for poly kernel
        print('----------------------------')
        print(f'Validação Cruzada ANN do {ativo}')
        f_measure_elm = []
        for idx in range(600, 840):#len(dados)-2
            f = annTestELM(self.dados, idx, act_fun_cv, num_hide_cv, c_cv, solucao_cv).getFmeasure()
            f_measure_elm.append(f)
            #print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_elm_mean = np.average(f_measure_elm)
        print(f'F-Measure médio: {f_measure_elm_mean}')
        
        plt.figure()
        plt.plot(range(600, 840), f_measure_elm, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation for ANN - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('ANN_' + ativo)
        #plt.show()
####################################################################
#XGBoost
####################################################################
    def testParametrosXGBoost(self, salvar, nome_arquivo):
        f_measure_parameters = pd.DataFrame(columns=['Solucao', 'ActFun', 'NumHiden', 'C', 'Fmeasure'])

        print(f'Booster     ETA  ColSample  Lambda    Alpha  Estimator Depth    SubSample   F-Measure')
        for booster_type_test in ['gbtree', 'dart', 'gblinear']:
            for eta_value_test in [0.1, 0.25, 0.5, 0.75, 0.9]:
                for colsample_value_test in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    for lambda_value in [0.1, 0.5, 1, 5, 10, 50, 100]:
                        for alpha_value_test in [0.1, 0.5, 1, 5, 10, 50, 100]:
                            for estimators_value_test in [1, 5, 10, 15, 20, 30]:
                                for depth_value_test in [1, 5, 10, 20, 30, 50, 100]:
                                    for subsample_value_test in [0.1, 0.25, 0.5, 0.75, 0.9]:
                                        f = xgBoostTest(self.dados, 600, booster_type_test, eta_value_test, colsample_value_test, 
                                                        lambda_value, alpha_value_test, estimators_value_test, depth_value_test, 
                                                        subsample_value_test).getFmeasure()
                                        f_measure_parameters = f_measure_parameters.append(
                                            {"Booster": booster_type_test,"ETA": eta_value_test, 
                                                "ColSample": colsample_value_test, "Lambda": lambda_value, 
                                                "Alpha": alpha_value_test, "Estimator": estimators_value_test,
                                                "Depth": depth_value_test, "SubSample": subsample_value_test}, ignore_index=True)         
                                        print(f'{booster_type_test}  {eta_value_test}   {colsample_value_test}  {lambda_value}  {alpha_value_test}    {estimators_value_test} {depth_value_test}  {subsample_value_test}  {f}')

        if (salvar > 0):
            f_measure_parameters.to_csv(nome_arquivo)

        return f_measure_parameters.sort_values("Fmeasure", ascending=False)
    
####################################################################
#Ensable
#################################################################### 
    def valitationEnsamble(self, save, ativo, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann):
        #Cross-Valitation for poly kernel
        print('----------------------------')
        print(f'Validação Cruzada Ensamble do {ativo}')
        f_measure_ensamble = []
        f_measure_poly = []
        f_measure_rbf = []
        f_measure_ann = []
        for idx in range(600, 840):#len(dados)-2
            h = ensableTest(self.dados, idx, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann)
            f_measure_ensamble.append(h.getFmeasure())
            f_measure_poly.append(h.getFmeasurePoly())
            f_measure_rbf.append(h.getFmeasureRbf())
            f_measure_ann.append(h.getFmeasureAnn())
            print(f'Size Train: {idx} - Ensamble: {h.getFmeasure()} Poly: {h.getFmeasurePoly()}    Rbf: {h.getFmeasureRbf()} ANN: {h.getFmeasureAnn()}')

        f_measure_ensamble_mean = np.average(f_measure_ensamble)
        print(f'F-Measure médio Ensamble: {f_measure_ensamble_mean}')
        f_measure_poly_mean = np.average(f_measure_poly)
        print(f'F-Measure médio Poly: {f_measure_poly_mean}')
        f_measure_rbf_mean = np.average(f_measure_rbf)
        print(f'F-Measure médio Rbf: {f_measure_rbf_mean}')
        f_measure_ann_mean = np.average(f_measure_ann)
        print(f'F-Measure médio ANN: {f_measure_ann_mean}')

        plt.figure()
        plt.plot(range(600, 840), f_measure_ensamble, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation for Ensable - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('Ensamble_' + ativo)
        #plt.show()

        plt.figure()
        plt.plot(range(600, 840), f_measure_poly, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation for SVM Polynomial - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('SVM_poly_' + ativo)
        #plt.show()


        plt.figure()
        plt.plot(range(600, 840), f_measure_rbf, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation for SVM Radial - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('SVM_rbf' + ativo)
        #plt.show()

        plt.figure()
        plt.plot(range(600, 840), f_measure_ann, linewidth=0.5, color='b')
        plt.xlabel('Cros-Valitation Iteration (Num Samples on Traning)')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation for ANN - ' + ativo)
        #plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig('ANN_' + ativo)
        #plt.show()

####################################################################
#Cross Valitation
#################################################################### 
    def crossValitation(self, save, ativo, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann):
        #Cross-Valitation for poly kernel
        print('----------------------------')
        print(f'Validação Cruzada Ensamble do {ativo}')
        acertos_ensamble = 0
        acertos_ensamble_unanime = 0
        duvida_ensamble_unanime = 0
        acertos_svm_poly = 0
        acertos_svm_rbf = 0
        acertos_ann = 0

        inicio = 600
        tamanho = 600 

        for idx in range(inicio, inicio+tamanho):
            acertos = testD1(self.dados, idx, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann).getAcertos()
            if acertos[0] == 1:
                acertos_ensamble = acertos_ensamble + 1
            if acertos[1] == 1:
                acertos_ensamble_unanime = acertos_ensamble_unanime + 1
            if acertos[1] == -1:
                duvida_ensamble_unanime = duvida_ensamble_unanime + 1
            if acertos[2] == 1:
                acertos_svm_poly = acertos_svm_poly + 1
            if acertos[3] == 1:
                acertos_svm_rbf = acertos_svm_rbf + 1
            if acertos[4] == 1:
                acertos_ann = acertos_ann + 1
            print(f'Size Train: {idx}')

        
        print(f'F-Measure Cross Validation Ensamble: {acertos_ensamble/tamanho}')
        print(f'F-Measure Cross Validation Ensamble Unanime total: {acertos_ensamble_unanime/tamanho}')
        print(f'Dúvida no Ensamble Unanime: {duvida_ensamble_unanime}')
        print(f'F-Measure Cross Validation Ensamble: {acertos_svm_poly/tamanho}')
        print(f'F-Measure Cross Validation Ensamble: {acertos_svm_rbf/tamanho}')
        print(f'F-Measure Cross Validation ANN: {acertos_ann/tamanho}')

    def crossValitationSave(self, save, ativo, 
                 nu_poly, degree_poly, gamma_poly,
                 nu_rbf, gamma_rbf,
                 act_fun_ann, num_hide_ann, c_ann, solucao_ann):
        inicio = 1140
        tamanho = 60 

        with open("ativo_" + str(inicio) + ".csv", "w", newline="") as student_file:
            writer = csv.writer(student_file)
            

            for idx in range(inicio, inicio+tamanho):
                acertos = testD1(self.dados, idx, 
                    nu_poly, degree_poly, gamma_poly,
                    nu_rbf, gamma_rbf,
                    act_fun_ann, num_hide_ann, c_ann, solucao_ann).getAcertos()

                writer.writerow(acertos)


        
    
    


arquivo = 'Tudo_PETR4_1242_FROM_2018_11_30_TO_2023_11_30.csv'
handle = avaliacao(arquivo)
#handle.crossValitationSVM(1, 'PETR4', 'poly', 0.25, 1, 4.5)
#handle.crossValitationSVM(1, 'PETR4', 'rbf', 0.75, 3, 0.5)
#handle.crossValitationANN(1, 'PETR4', 'no_re', 'relu', 50, 0.7)

#handle.crossValitationEnsamble(1, 'PETR4', 0.25, 1, 4.5,
#                               0.75, 0.5, 
#                               'relu', 50, 0.7, 'no_re')

#handle.crossValitation(1, 'PETR4', 0.25, 1, 4.5,
#                       0.75, 0.5, 
#                       'relu', 50, 0.7, 'no_re')

arquivo = 'Tudo_VALE3_1242_FROM_2018_11_30_TO_2023_11_30.csv'
handle = avaliacao(arquivo)
#handle.crossValitationANN(1, 'VALE3', 'no_re', 'leaky_relu', 50, 0.4)
#handle.crossValitationSVM(1, 'VALE3', 'rbf', 0.25, 3, 1)
#handle.crossValitationSVM(1, 'VALE3', 'poly', 0.1, 3, 3)

#handle.valitationEnsamble(1, 'VALE3', 0.1, 3, 3,
#                               0.25, 1, 
#                               'leaky_relu', 50, 0.4, 'no_re')

#handle.crossValitation(1, 'VALE3', 0.1, 3, 3,
#                               0.25, 1, 
#                               'leaky_relu', 50, 0.4, 'no_re')


arquivo = 'Tudo_BBAS3_1242_FROM_2018_11_30_TO_2023_11_30.csv'

handle = avaliacao(arquivo)
#handle.crossValitationANN(1, 'BBAS3', 'no_re', 'sin', 20, 0.8)
#handle.crossValitationSVM(1, 'BBAS3', 'rbf', 0.25, 3, 10)#segundo maior
#handle.crossValitationSVM(1, 'BBAS3', 'poly', 0.25, 3, 4.5)#segundo maior (rodando paralelo)
#handle.crossValitationEnsamble(1, 'BBAS3', 
#                               0.25, 3, 4.5,
#                               0.25, 10,
#                               'sin', 20, 0.8, 'no_re')

#handle.crossValitation(1, 'BBAS3',
#                       0.25, 3, 4.5,
#                       0.25, 10,
#                       'sin', 20, 0.8, 'no_re')


arquivo = 'Tudo_ITUB4_1243_FROM_2018_11_28_TO_2023_11_30.csv'

handle = avaliacao(arquivo)
#handle.crossValitationANN(1, 'ITUB4', 'no_re', 'leaky_relu', 100, 0.1)
#handle.crossValitationSVM(1, 'ITUB4', 'rbf', 0.5, 3, 1.5)
#handle.crossValitationSVM(1, 'ITUB4', 'poly', 0.25, 3, 5)#segundo maior (rodando paralelo 2)

#handle.crossValitationEnsamble(1, 'ITUB4', 
#                               0.25, 3, 5,
#                               0.5, 1.5,
#                               'leaky_relu', 100, 0.1, 'no_re')

handle.crossValitation(1, 'ITUB4', 
                       0.25, 3, 5,
                       0.5, 1.5,
                       'leaky_relu', 100, 0.1, 'no_re')
