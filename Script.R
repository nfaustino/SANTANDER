#library("neuralnet")
library("party")
library("glmnet")
library("xgboost")
library("pROC")


setwd("C:/Users/faustnun/Desktop/Projects/_Kaggle/Santander")
set.seed(123)

# Load datasets
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Check if columns names match without the last collumn
colnames(test)==colnames(train[,1:370])

# Get the name of the 369 inputs
input_names <- colnames(train[2:370])

# Draw the formula
# a <- as.formula(paste('TARGET ~ ' ,paste(input_names,collapse='+')))
#
# # Run neural network with 10 hidden layers
# nnet <- neuralnet(a , train, hidden=10, threshold=0.01)
# 
# 
# nnet
# plot(nnet)
# View(nnet$result.matrix)
# 
# train_results <- compute(nnet,train[,2:370])
# 
# test_results <- compute(nnet,test[,2:370])
# 
# train_view <- cbind(train[,c("ID","TARGET")],as.data.frame(train_results$net.result))


#Elastic Net

net <- glmnet(as.matrix(train[,2:370]),train[,371:371], family="binomial", alpha=0, nlambda=1000,standardize=TRUE,maxit=10000000)

cvnet <- cv.glmnet(as.matrix(train[,2:370]),train[,371:371])

plot(net)
plot(cvnet)

netpred <-predict(net,as.matrix(train[,2:370]), type = "response", s = cvnet$lambda.min)

netpred <- as.data.frame(netpred) 



roccurve <- roc(train$TARGET~netpred[,c(1)])
plot(roccurve)


## optimal cut-off point 
cutoff <- roccurve$thresholds[which.max(roccurve$sensitivities + roccurve$specificities)]
#cutoff <- quantile(netpred[,c(1)],probs=0.95)
cutoff

netpred$pred <- as.numeric(netpred[,c(1)])
netpred$pred[netpred[,c(1)]>cutoff]  <- 1
netpred$pred[netpred[,c(1)]<=cutoff] <- 0

# Area Under the Curve
auc(roccurve) # auc(train$TARGET,netpred[,c(1)])
# Mean Error
mean(netpred$pred!=train$TARGET)
# Confusion Table
xtabs(~ train$TARGET + netpred$pred)

cor(netpred[,c(1)],train$TARGET)^2
cor(netpred$pred,train$TARGET)^2


coeffs<-coef(net, s=cvnet$lambda.min)
coeffs<-as.data.frame(cbind(coeffs@Dimnames[[1]], as.numeric(coeffs)))



# Classification Trees
input_names2 <- coeffs[coeffs$V2!=0&coeffs$V1!="(Intercept)",c(1)]


b <- as.formula(paste('TARGET ~ ' ,paste(input_names2,collapse='+')))

b

ct = ctree(b,train)

summary(ct)
plot(ct)

cor(predict(ct),train$TARGET)^2

rocct<- roc(train$TARGET~predict(ct))

plot(rocct)


cutoffct <- rocct$thresholds[which.max(rocct$sensitivities + rocct$specificities)]
cutoff


ctpred <-as.data.frame( predict(ct))
ctpred$pred <- ctpred$TARGET
ctpred$pred[ctpred$TARGET>cutoffct]  <- 1
ctpred$pred[ctpred$TARGET<=cutoffct] <- 0

# Area under the curve
auc(rocct)
# Mean Error
mean(ctpred$pred!=train$TARGET)
# Confusion Matrix
xtabs(~ train$TARGET + ctpred$pred)

names(test[,c(as.vector(input_names2))])

#preds <- predict(ct, test[,c(as.vector(input_names2))])
testpred <- as.data.frame(predict(net,as.matrix(test[,2:370]), type = "response", s = cvnet$lambda.min))
testpred$pred[testpred[,c(1)]>cutoff]  <- 1
testpred$pred[testpred[,c(1)]<=cutoff] <- 0

submission <- data.frame(ID=test$ID, TARGET=testpred$pred)

# XGboost

train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
  return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
  if (length(unique(train[[f]])) == 1) {
    cat(f, "is constant in train. We delete it.\n")
    train[[f]] <- NULL
    test[[f]] <- NULL
  }
}

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(train[[f1]] == train[[f2]])) {
      cat(f1, "and", f2, "are equals.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]

#---limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
for(f in colnames(train)){
  lim <- min(train[,f])
  test[test[,f]<lim,f] <- lim
  
  lim <- max(train[,f])
  test[test[,f]>lim,f] <- lim  
}
#---

train$TARGET <- train.y


train_m <- sparse.model.matrix(TARGET ~ ., data = train)

dtrain <- xgb.DMatrix(data=train_m, label=train.y)
watchlist <- list(train_m=dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.0202048,
                max_depth           = 5,
                subsample           = 0.6815,
                colsample_bytree    = 0.701
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 560, 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)


test$TARGET <- -1

test_m <- sparse.model.matrix(TARGET ~ ., data = test)

preds <- predict(clf, test_m)
submission_xgb <- data.frame(ID=test.id, TARGET=preds)

xgbpred <- predict(clf,train_m)
roccurve_xgb <- roc(train$TARGET~xgbpred)

## optimal cut-off point 
cutoff_xgb <- roccurve_xgb$thresholds[which.max(roccurve_xgb$sensitivities + roccurve_xgb$specificities)]

xgbpred <- data.frame(xgbpred)
xgbpred$pred[xgbpred$xgbpred>cutoff_xgb]  <- 1
xgbpred$pred[xgbpred$xgbpred<=cutoff_xgb] <- 0

plot(roccurve_xgb)

# Area Under the Curve
auc(roccurve_xgb)
# Mean Error
mean(xgbpred$pred!=train$TARGET)
# Confusion Matrix
xtabs(~ train$TARGET + xgbpred$pred)

var_imp <- xgb.importance(colnames(train_m),model=clf)

xgb.plot.importance(var_imp[1:10])

cat("saving the submission file\n")
write.csv(submission_xgb, "submission.csv", row.names = F)

#########

dcast(train,"var15"~TARGET,value.var='var15',mean)

hist(train$var15[train$TARGET==1])
hist(train$var15[train$TARGET==0],add=T)
boX()


boxplot(train$var15~train$TARGET)
