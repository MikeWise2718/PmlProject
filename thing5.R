library(data.table)
library(ElemStatLearn)
library(randomForest)
library(caret)

# Data Processing
## Load the data

otrn <- data.table(read.csv("pml-training.csv"))
otst <- data.table(read.csv("pml-testing.csv"))

## Get rid of all non-numeric columns not present in both datasets
cnstst <- colnames(otrn)[7:59]
lntrn <- length(otrn)
lntst <- length(otst)

ntrn <- data.table(user_name=otrn[["user_name"]],classe=otrn[["classe"]])
ntst <- data.table(user_name=otst[["user_name"]])

for (i in 1:length(cnstst))
{
  cn <- cnstst[[i]]
  if (cn=="num_window") next
  clstst <- class(otst[[cn]])
  clstrn <- class(otrn[[cn]])
  if (clstst!="numeric" && clstst !="integer")
  {
    print(sprintf("%d tst %s is not numeric/integer but %s",i,cn,clstst))
    next
  }
  if (clstrn!="numeric" && clstrn !="integer")
  {
    print(sprintf("%d trn %s is not numeric/integer but %s",i,cn,clstrn))
    next
  }
  ntrn[[cn]] <- otrn[[cn]]
  ntst[[cn]] <- otst[[cn]]
}

# Check the quality

ona <- sum(is.na(otrn))
nna <- sum(is.na(ntrn))
msg <- sprintf("Original training na count:%d  - After processing:%d",ona,nna)
print(msg)

# Model Fitting

## Random Forests
set.seed(2718)
rffit <- randomForest(classe ~ ., ntrn, importance=T)

prftrn <- predict(rffit, ntrn)
confusionMatrix(prftrn, ntrn$classe)

varImpPlot(rffit)

prftst <- predict(rffit, ntst)
prftst

## Boost Trees
set.seed(2718)
btfit <- train(classe ~ ., method="gbm", data=ntrn, verbose=F)

pbttrn <- predict(btfit)
confusionMatrix(pbttrn, ntrn$classe)

pbttst <- predict(btfit, ntst)
pbttst