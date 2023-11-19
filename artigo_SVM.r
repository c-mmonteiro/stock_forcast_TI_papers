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

dados<-read.table(file="TI_Tudo_PETR4_2520_FROM_2018_09_28_TO_2023_09_28.csv",header=TRUE,sep=",",dec=".")


dados <- subset(dados, select = -X)
dados <- subset(dados, select = -time)
dados <- subset(dados, select = -open)
dados <- subset(dados, select = -high)
dados <- subset(dados, select = -low)
dados <- subset(dados, select = -real_volume)

head(dados)

svmTest <- function(dados_fnc, n_train, kernel_type){
  #kernel type: "polynomial" or 
  
  # Definindo o conjunto de treinamento e teste
  train <- dados_fnc[row.names(dados_fnc) %in% 1:n_train, ]
  train_bruto <- subset(train, select = -direcao)
  train_resultado <- train$direcao
  
  test<- dados_fnc[row.names(dados_fnc) %in% (n_train+1):nrow(dados_fnc), ]
  test_bruto <- subset(test, select = -direcao)
  test_resultado <- test$direcao

  
  # Criando modelo SVM a partir do conjunto de dados
  modelo_svm <- svm(direcao ~ ., data = train, type = 'nu-classification', kernel = kernel_type)
  # Aplica o modelo na previsão
  previsao <- predict(modelo_svm, test_bruto)
  
  tab_v1 <- table(previsao, test_resultado)
  
  f_measure <- (tab_v1[1,1] + tab_v1[2,2])/length(previsao)
  

  return(f_measure)
}

#Cross Valitation for poly kernel
f_measure_poly <- c()
for (idx in 600:840){#nrow(dados)-3
  f <- svmTest(dados, idx, "polynomial")
  f_measure_poly <- c(f_measure_poly, f)
  print(paste("Tamanho do Treino: ", idx, " - F-Measure: ", f))
}
f_measure_poly_mean <- mean(f_measure_poly)
sprintf("F-Measure médio: %f", f_measure_poly_mean)

#Cross Valitation for radial kernel
f_measure_radial <- c()
for (idx in 600:840){
  f <- svmTest(dados, idx, "radial")
  f_measure_radial <- c(f_measure_radial, f)
  print(paste("Tamanho do Treino: ", idx, " - F-Measure: ", f))
}
f_measure_radial_mean <- mean(f_measure_radial)
sprintf("F-Measure médio RBF: %f", f_measure_radial_mean)
sprintf("F-Measure médio Poly: %f", f_measure_poly_mean)



plot(f_measure_poly, 
     type = "l",
     col="black",
     xlab = "Interação do Cros-Valitation Iniciando em 50%", 
     ylab = "F-Measure", 
     main = "F-Measure evolution on Cross-Valitation - R version",
     ylim = c(0.1, 0.9))
lines(f_measure_radial, 
     type = "l",
     col="red")
legend("topleft",
       lty = 1,
       legend = c("Poly", "Radial"),
       col = c("black", "red"),
       ncol = 1)