import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from algoritimos import *  
from indicadores_tecnicos import *  


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
    
    def crossValitationSVM(self, save, fig_name):
        #Cross-Valitation for poly kernel
        f_measure_poly = []
        for idx in range(600, 840):#len(dados)-2
            f = svmTest(self.dados, idx, 'poly', 0.75, 1, 4.5).getFmeasure()
            f_measure_poly.append(f)
            print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_poly_mean = np.average(f_measure_poly)
        print(f'F-Measure médio: {f_measure_poly_mean}')

        #Cross-Valitation for radial kernel
        f_measure_radial = []
        for idx in range(600, 840):
            f = svmTest(self.dados, idx, 'rbf', 0.90, 3, 10.0).getFmeasure()
            f_measure_radial.append(f)
            print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_radial_mean = np.average(f_measure_radial)
        print(f'F-Measure médio RBF: {f_measure_radial_mean}')
        print(f'F-Measure médio Poly: {f_measure_poly_mean}')



        plt.plot(f_measure_poly, linewidth=0.5, color='b', label='Poly')
        plt.plot(f_measure_radial, linewidth=0.5, color='r', label='Radial')
        plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation')
        plt.legend(title='Kernel:')
        if (save > 0):
            plt.savefig(fig_name)
        plt.show()

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
    
    def crossValitationANN(self, save, fig_name):
        #Cross-Valitation for poly kernel
        f_measure_elm = []
        for idx in range(600, 840):#len(dados)-2
            f = annTestELM(self.dados, idx, 'sigmoid', 10, 0.4, 'no_re').getFmeasure()
            f_measure_elm.append(f)
            print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_elm_mean = np.average(f_measure_elm)
        print(f'F-Measure médio: {f_measure_elm_mean}')

        plt.plot(f_measure_elm, linewidth=0.5, color='b', label='ELM')
        plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation - ELM')
        if (save > 0):
            plt.savefig(fig_name)
        plt.show()
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
    
    def crossValitationANN(self, save, fig_name):
        #Cross-Valitation for poly kernel
        f_measure_elm = []
        for idx in range(600, 840):#len(dados)-2
            f = annTestELM(self.dados, idx, 'sigmoid', 10, 0.4, 'no_re').getFmeasure()
            f_measure_elm.append(f)
            print(f'Tamanho do Treino: {idx} - F-Measure: {f}')

        f_measure_elm_mean = np.average(f_measure_elm)
        print(f'F-Measure médio: {f_measure_elm_mean}')

        plt.plot(f_measure_elm, linewidth=0.5, color='b', label='ELM')
        plt.xlabel('Interação do Cros-Valitation Iniciando em 50%')
        plt.ylabel('F-Measure')
        plt.title('F-Measure evolution on Cross-Valitation - ELM')
        if (save > 0):
            plt.savefig(fig_name)
        plt.show()



arquivo = 'Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv'

avaliacao(arquivo).testParametrosXGBoost(1, 'parametros_xgboost.csv')

################################################
####            Teste de Parametros
#parametros = avaliacao(arquivo).testParametrosSVM(0,0)
#print(parametros[parametros["Kernel"] == "poly"].head())
#print(parametros[parametros["Kernel"] == "rbf"].head())

#avaliacao(arquivo).crossValitationSVM()
