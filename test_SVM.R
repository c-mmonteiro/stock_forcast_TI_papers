#Limpeza da Memória
rm(list=ls(all=TRUE))

#Definindo o conjunto de dados
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Links úteis
#https://rpubs.com/amrofi/conceitos_time_series_intro2
#https://rstudio-pubs-static.s3.amazonaws.com/254962_6d87741d9ad64a32b6e47d684875dde9.html

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
library("e1071")
library(plotly)

dados<-read.table(file="SVM_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv",header=TRUE,sep=",",dec=".")


head(dados)



dados <- subset(dados, select = -X)
 
 
# Definindo o conjunto de treinamento e teste
n <- 1000
train <- dados[row.names(dados) %in% 1:n, ]
train_bruto <- subset(train, select = -direcao)
train_resultado <- train$direcao

test<-dados[row.names(dados) %in% (n+1):nrow(dados), ]
test_bruto <- subset(test, select = -direcao)
test_resultado <- test$direcao


# Criando modelo SVM a partir do conjunto de dados iris
modelo_svm <- svm(direcao ~ ., data = train, type = 'nu-classification', kernel = "polynomial")

for (i in 1:nrow(test)){
  i
  if (i > 1){
    train[nrow(train) + 1,] <- test[i,]
    modelo_svm <- svm(direcao ~ ., data = train, type = 'nu-classification', kernel = "polynomial")
    teste01_linha <- predict(modelo_svm, test_bruto[i,])
    teste01 <- c(teste01, list(teste01_linha))
  }
  else{
    teste01 <- predict(modelo_svm, test_bruto[i,])
  }
  
}
# Resumo do modelo
#summary(modelo_svm)


teste001 <- predict(modelo_svm, test_bruto)
#tabela V1
table(teste001, test_resultado)
#tabela V2
table(teste01, test_resultado)

test_resultado_f <- factor(test_resultado)

line_b <- list(
   type = "line",
   xref = 'x',
   yref = 'paper',
   y0 = 0,
   y1 = 1,
   line = list(color = 'black',
   width = 0.5))

line_r <- list(
  type = "line",
  xref = 'x',
  yref = 'paper',
  y0 = 0,
  y1 = 1,
  line = list(color = 'red',
              width = 0.5))
 
lines <- list()
 erro_v1 <- 0
 erro_v2 <- 0
 for(i in 1:length(teste001)){
   if(teste001[i] != test_resultado_f[i]){
     line_b[["x0"]] <- i
     line_b[["x1"]] <- i
     lines <- c(lines, list(line_b))
     erro_v1 <- erro_v1 + 1
   }
   
   if(teste01[i] != test_resultado_f[i]){
     line_r[["x0"]] <- i
     line_r[["x1"]] <- i
     lines <- c(lines, list(line_r))
     erro_v2 <- erro_v2 + 1
   }
   
 }
#Erro com apenas 1 treinamento
erro_v1/length(teste001)
#Erro com vários treinamentos
erro_v2/length(teste01)

fig <- test_bruto %>% plot_ly(type="candlestick",
                      open = ~open1, close = ~close1,
                      high = ~high1, low = ~low1) 
fig <- fig %>% layout(title = "Basic Candlestick Chart",
                     shapes = lines)
fig