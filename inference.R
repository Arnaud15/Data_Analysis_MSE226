library(tidyverse)

df_train = read.csv("CleanedData/Features1_train.csv")
df_test = read.csv("CleanedData/Features1_test.csv")
df_train$X <- NULL
df_test$X <- NULL

model = lm(log_target ~ ., data=df_train)
