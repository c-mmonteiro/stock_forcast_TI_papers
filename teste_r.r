#Limpeza da Memória
rm(list=ls(all=TRUE))

#Definindo o conjunto de dados
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


#Pacotes Necessários
library(forecast)  #Principal pacote para previsão
library(tsibble)   #Para gerar objetos tipo tisbble e usar nas previsões
library(fable)     #Para usar a função model()
library(lubridate) #Para inserir datas
library(dplyr)     #Para operadores como mutate
library(ggplot2)   #Para gerar gráficos
library(feasts)    #Pacote para características em séries temporais
library(reshape2)
library(fpp3)
library(urca)
library(trend) #Para testes de Man-Kendall e Cox-Stuart 
#https://cran.r-project.org/web/packages/trend/index.html

#Lendo um conjunto de dados externo ao R

dados<-read.table(file="Preco_PETR4_1240_FROM_2018_09_10_TO_2023_09_08.csv",header=TRUE,sep=",",dec=".")

head(dados)     #Mostrando as seis primeiras linhas do conjunto
tail(dados,12)  #Mostrando as doze últimas linhas do conjunto

#Adicionando os meses para estudo
dados$time <- as.Date(dados$time)



#Convertendo para um objeto tsibble

dados.ts <- tsibble(
  Data = dados$time,
  preco = dados$close,
  index = Data
)

dados.ts<-dados.ts %>%
  as_tsibble(index = Data) %>%
  group_by_key(dados.ts) %>%
  fill_gaps() %>%
  tidyr::fill(preco, .direction = "down")

dados.ts['preco']

dts <-ts(dados['close'])

#Gráfico 1
x11()
g<-autoplot(dados.ts,xlab="Tempo (diário)",ylab="Preço (R$)")+
  theme_bw() 

#Para fazer gráfico iterativo
library(plotly)
x11()
ggplotly(g)

################################################################################
#########################     AULA 2    ########################################
################################################################################



#################################Teste ADF######################################

#H0: Série tem raiz unitária, então NÃO é estacionária
#Ha: Série NÃO tem raiz unitária, então é estacionária

args(ur.df)   #Mostra os argumentos da função que computa o teste ADF

adf_test<-ur.df(y=dts,lags=5,type="trend",selectlags="AIC")
adf_test@testreg

#z.lag.1 é o coeficiente de interesse para teste da raiz unitária.
#e para avaliar a sua insignificância precisamos da tabela de valores
#críticos que fica na variável adf_test@cval do teste.

#Estatística do teste
adf_test@cval 
summary(adf_test)@teststat

#Como o valor t para z.lag.1  = -9.036 e o valor de tau3 em 5% é -3.43,
#ao avaliar |-9.036| > |-3.43|, rejeita-se H0. Então a série não tem raiz
#unitária, portanto é estaticonária.

#Análise dos resíduos da regressão
plot(adf_test)

#################################Teste KPSS######################################

#H0: Série NÃO tem raiz unitária, então É estacionária
#Ha: Série tem raiz unitária, então NÃO é estacionária


#O teste é definido a partir de:
#yt = dt+rt+et, em que dt é componente de tendência, rt de random walk e et um erro.

args(ur.kpss)

kpss_test<-ur.kpss(dts,type="tau",lags="short")
summary(kpss_test) 

#Comparando o valor da estatística do teste com o valor crítico para 5%
# temos que |0.022| < |0.146| nãO se rejeita a H0, então é série NÃO tem raiz
#unitária, logo É estacionária.

############################Teste Run###########################################
library(randtests) #Para o teste Run
#https://cran.r-project.org/web/packages/randtests/index.html


#H0: Sequência gerada aleatoriamente, então NÃO apresenta tendência
#Ha: Sequência NÃO FOI gerada aleatoriamente, então apresenta tendência

args(runs.test)

runs_test<-runs.test(dts)
runs_test

#Considerando um nível de significância de 5%, com p-valor=4.70e-13, rejeita H0
#então a série apresenta tendência.

####IMPORTANTE####
#O Runs Test também nos informou que a componente tendência estava presente
#p-valor = 0.02146,o teste de Runs é mais rigoroso se em uma parte dos dados
#for observado uma subida ou uma descida ele irá acusar
#tendência por ele ser um teste para aleatoridade

############################Teste Man-Kendall###################################

#H0: NÃO apresenta tendência
#Ha: Apresenta tendência

mankendall_test<-mk.test(dts)
mankendall_test

##############################Teste Cox-Stuart##################################

#H0: NÃO apresenta tendência
#Ha: Apresenta tendência

cs_test<-cox.stuart.test(dts)
cs_test

############################Teste de Kruskal-Wallis#############################
#Para este teste, cada mês é suposto como uma amostra de uma população

#H0: NÃO apresenta sazonalidade
#Ha: Apresenta sazonalidade

kw_test<-kruskal.test(dados$close ~ dados$time)
kw_test

#Considerando um nível de significância de 5%, com p-valor=0.4856, 
#NÃO rejeita H0 então a série NÃO apresenta sazonalidade.

############################Autocorrelação (ACF) ###############################
ggAcf(dados[,3])
#A ACF gerada sugere a presença de ciclos nos dados.

ggPacf(dados[,3])

################################################################################
#########################     AULA 3    ########################################
################################################################################





# Definindo o conjunto de treinamento 2003 to 2006
train <- dados.ts %>%
  filter_index("2021-09-01" ~ "2023-01-01")

test<-dados.ts %>%
  filter_index("2023-01-02" ~ "2023-09-01") 

# Ajustando os modelos
temperatura.fit <- train %>%
  model(
    Mean = MEAN(preco),
    `Naïve` = NAIVE(preco),
    `Seasonal naïve` = SNAIVE(preco),
    Drift = NAIVE(preco ~ drift()),
    'Arima' = fable::ARIMA(preco)
  )
temperatura.fit


# Gerando previsão para os 12 meses seguintes

temperatura.fc <-temperatura.fit %>% forecast(h = 180)

# Plotando os valores atuais e as previsões

x11()
temperatura.fc %>%
  autoplot(train, level = NULL) +
  autolayer(
    filter_index(dados.ts, "2021-09-01" ~ .),
    colour = "black"
  ) +
  labs(
    y = "Graus Celsius",
    title = "Temperatura Mensal no RJ"
  ) +
  guides(colour = guide_legend(title = "Previsão"))


#Análisando os resíduos das previsões

#Neste objeto estará três importante informações
#.fitted contém os valores ajustados;
#.resid contém os resíduos;
#.innov contém os “resíduos inovadores” que, neste caso,
# são idênticos aos resíduos regulares.

resid.fit<-augment(temperatura.fit)
resid.fit

View(resid.fit)

#ACF para os resíduos do modelo Naive
resid.fit %>% 
  filter(.model == "Seasonal naïve") %>%
  ACF(.innov) %>%
  autoplot() +
  labs(title = "Resíduos obtidos com o método naïve")

#Histograma para os resíduos do modelo Naive
resid.fit %>% 
  filter(.model == "Seasonal naïve") %>%
  ggplot(aes(x = .innov)) +
  geom_histogram() +
  labs(title = "Histograma dos resíduos")

#Resíduos ao longo do tempo
resid.fit %>% 
  filter(.model == "Seasonal naïve") %>%
  autoplot(.innov) +
  labs(y = "Temperatura",
       title = "Resíduos obtidos com o método naïve")


#Fazer para os outros dois modelos

#Testes para análise da Correlação
#H0: Correlações iguais a zero (Não tem autocorrelação)
#Ha: Pelo menos uma correlação diferente de zero

resid.fit %>% 
  filter(.model == "Seasonal naïve") %>%
  features(.innov, box_pierce, lag = 10, dof = 0)

resid.fit %>% 
  filter(.model == "Seasonal naïve") %>%
  features(.innov, ljung_box, lag = 10, dof = 0)

#Ano nível de 1% de confiança p-valor > 1%, então não rejeita H0, então
#os resíduos não são correlacionados.


#Medidas de acurácia

acc.models<- fabletools::accuracy(temperatura.fc, dados.ts) %>%
  mutate(Method = paste(.model, "method")) %>%
  select(Method, RMSE, MAE, MAPE)
acc.models


#########################Validação Cruzada##############################
#Ref: https://rpubs.com/amrofi/cross_validation_fable 
#Ref: https://stackoverflow.com/questions/64863228/cross-validation-of-monthly-time-series-using-fable-package
#Ref: https://bookdown.org/mpfoley1973/time-series/exponential.html#model-selection-with-cv

#Argumentos para validação cruzada usando tsibble
#stretch_tsibble() --> Usada para criar vários conjuntos de treinamento
#.init=3'          --> Número de elementos do conjunto inicial
#.step=1           --> Aumento da dimensão dos conjuntos de formação sucessivos 

#Criando os conjuntos para Validação Cruzada
dados.ts.tr <- dados.ts %>%
  stretch_tsibble(.init = 12, .step = 1) 
dados.ts.tr

# Accuracia
fc <- dados.ts.tr %>%
  model(`Seasonal Naive` = SNAIVE(Temperatura)) %>%
  forecast(h = "1 year") %>%
  group_by(.id) %>%
  mutate(h = row_number()) %>%
  ungroup() %>%
  as_fable(response="Temperatura", distribution=Temperatura)

fc %>%
  fabletools::accuracy(dados.ts, by=c("h",".model")) %>%
  mutate(Method = paste(.model, "method")) %>%
  select(Method, RMSE, MAE, MAPE)


#Para verificar outra maneira de realizar Validação Cruzada em TS, consultar
# #Ref: https://robjhyndman.com/hyndsight/tscv-fable/ 

#Outra maneira de fazer a validação cruzada
#Ref: https://robjhyndman.com/hyndsight/tscv/ 




