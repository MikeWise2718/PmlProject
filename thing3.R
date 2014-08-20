library(data.table)
library(ElemStatLearn)
library(randomForest)
library(caret)

otrn <- data.table(read.csv("pml-training.csv"))
otst <- data.table(read.csv("pml-testing.csv"))

cnstst <- colnames(otrn)[7:59]
cnstrn <- c()
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
  cnstrn <- c(cnstrn,cn)
  ntrn[[cn]] <- otrn[[cn]]
  ntst[[cn]] <- otst[[cn]]
}

summary(ntrn)

ona <- sum(is.na(otrn))
nna <- sum(is.na(ntrn))
msg <- sprintf("Original training na count:%d  - After processing:%d",ona,nna)
print(msg)


set.seed(2718)
fit <- randomForest(classe ~ ., ntrn, importance=T)

ptrn <- predict(fit, ntrn)
confusionMatrix(ptrn, ntrn$classe)

varImpPlot(fit)

ptst <- predict(fit, ntst)
ptst