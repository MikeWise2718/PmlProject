library(data.table)
library(ElemStatLearn)
library(randomForest)
library(caret)
library(reshape)

# Data Processing
## Load the data

otrn <- data.table(read.csv("pml-training.csv"))
otst <- data.table(read.csv("pml-testing.csv"))

ggplot(otrn,aes(x=classe))+geom_histogram(fill=c("darkgreen","darkred","darkred","darkred","darkred")) +
       ggtitle("Participants and Exercise Grade") + facet_grid( . ~ user_name ) +
       labs(x="Grade/Class",y="Count")

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
  if (cn=="user_name") next
  clstst <- class(otst[[cn]])
  clstrn <- class(otrn[[cn]])
  if (clstst!="numeric" && clstst !="integer") next
  if (clstrn!="numeric" && clstrn !="integer") next
  ntrn[[cn]] <- otrn[[cn]]
  ntst[[cn]] <- otst[[cn]]
}

# Check the quality


hdat <- melt(ntrn)
ggplot(hdat,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()


ona <- sum(is.na(otrn))
nna <- sum(is.na(ntrn))
msg <- sprintf("Original training na count:%d  - After processing:%d",ona,nna)
print(msg)

# now split into training (70 percent) and valdiation (30 percent) sets
trnidx <- createDataPartition(y=ntrn$classe,p=0.8,list=F)
ntrndf <- data.frame(ntrn)
nvld <- ntrndf[-trnidx,]
ntrn <- ntrndf[trnidx,]

# 
# ggplot(ntrn,aes(x=ntrn$yaw_belt,y=ntrn$roll_arm))+geom_point(aes(col=ntrn$classe)) +
#   ggtitle("a title") 
# 
# ggplot(ntrn,aes(x=ntrn$yaw_belt,y=ntrn$roll_arm))+geom_point(aes(col=ntrn$user_name)) +
#   ggtitle("a title") 


# Model Fitting

## Random Forests
#set.seed(2718)
rffit <- randomForest(classe ~ ., ntrn, importance=T)
#rffit <- randomForest(classe ~ yaw_belt + pitch_belt + roll_belt + roll_arm + total_accel_arm + yaw_arm, ntrn, importance=T)

prftrn <- predict(rffit, ntrn)
confusionMatrix(prftrn, ntrn$classe)

prfvld <- predict(rffit, nvld)
confusionMatrix(prfvld, nvld$classe)

varImpPlot(rffit)

prftst <- predict(rffit, ntst)
prftst

#deparse(do.call(call, as.list(c("c", as.character(prftst)))))

do.call(call, as.list(c("c", as.character(prftst))))