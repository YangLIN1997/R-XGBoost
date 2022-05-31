library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
setwd("/Users/yanglin/Desktop/R")

df <- read_csv("iris.csv")
df$variety<-factor(df$variety, levels = c("Setosa", "Versicolor", "Virginica"))
#standardize first four columns of iris dataset
#df[,1:4] <- as.data.frame(scale(df[,1:4]))
parts = createDataPartition(df$variety, p = 0.9, list = F)
train = df[parts, ]
test = df[-parts, ]

x_train = data.matrix(train[,-5])                 
y_train = train$variety                               

x_test = data.matrix(test[,-5])                  
y_test = test$variety                            

# one-hot-encoding categorical features
ohe_feats = c('Setosa', 'Virginica', 'Versicolor')

# convert the train and test data into xgboost matrix type.
xgboost_train = xgb.DMatrix(data=x_train, label=y_train)
xgboost_test = xgb.DMatrix(data=x_test, label=y_test)


# train a model using our training data
model <- xgboost(data = xgboost_train,                    # the data   
                 max.depth=3,                            # max depth 
                 nrounds=50)                              # max number of boosting iterations

summary(model)

#use model to make predictions on test data
pred_test = predict(model, xgboost_test)

pred_test


pred_test[(pred_test>3)] = 3
levels(y_test)
pred_y = as.factor((levels(y_test))[round(pred_test)])
print(pred_y)


conf_mat = confusionMatrix(y_test, pred_y)
print(conf_mat)



