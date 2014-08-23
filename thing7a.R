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
 # if (cn=="user_name") next
  clstst <- class(otst[[cn]])
  clstrn <- class(otrn[[cn]])
  if (clstst!="numeric" && clstst !="integer") next
  if (clstrn!="numeric" && clstrn !="integer") next
  ntrn[[cn]] <- otrn[[cn]]
  ntst[[cn]] <- otst[[cn]]
}

# Check the quality

ona <- sum(is.na(otrn))
nna <- sum(is.na(ntrn))
msg <- sprintf("Original training na count:%d  - After processing:%d",ona,nna)
print(msg)
ntrn0 <- ntrn

# Iterate
baseprobseq <- seq( 0.05, 0.95, variable0.05 )
#baseprobseq <- seq( 0.1, 0.9, 0.2 )
nsamp <- 3

pvec <- c()
avec <- c()
tvec <- c()
evec <- c()

for (i in 1:20)
{
  vcmd <- sprintf("v%d = c()",i)
  eval(parse(text=vcmd))
}

ntodo <- length(baseprobseq)*nsamp
idone <- 0
iprobdone <- 0
eta <- 0
acc <- 0
sttime <- proc.time()
for (prob in baseprobseq)
{
   for (j in 1:nsamp)
   {
      jsttime <- proc.time()
      celap <- (jsttime-sttime)["elapsed"]
      if (idone>0)
      {
          eta <- ntodo*celap/idone
      }
      msg <- sprintf("it:%d/%d prob:%5.2f last-acc:%5.3f elap:%6.1f eta-sec:%6.1f",idone,ntodo,prob,acc,celap,eta)
     # msg <- sprintf("it:%d/%d prob:%5.2f",idone,ntodo,prob)
      print(msg)

      trnidx <- createDataPartition(y=ntrn0$classe,p=prob,list=F)
      ntrndf <- data.frame(ntrn0)
      nvld <- ntrndf[-trnidx,]
      ntrn <- ntrndf[trnidx,]

      #rffit <- randomForest(classe ~ ., ntrn, importance=T)
      rffit <- randomForest(classe ~ yaw_belt + pitch_belt + roll_belt, ntrn, importance=T)
      #rffit <- randomForest(classe ~ yaw_belt + pitch_belt + roll_belt + roll_arm + total_accel_arm + yaw_arm, ntrn, importance=T)
     
      prfvld <- predict(rffit, nvld)
      cm <- confusionMatrix(prfvld, nvld$classe)
      acc <- cm$overall["Accuracy"]
      pvec <- c(pvec,prob)
      avec <- c(avec,acc)
      prftst <- predict(rffit, ntst)
      for (i in 1:20)
      {
        vcmd1 <- sprintf("vtmp = as.character(prftst[[%d]])",i)
        eval(parse(text=vcmd1))
        vcmd2 <- sprintf("v%d = c(v%d,vtmp)",i,i) 
        eval(parse(text=vcmd2))
      }
      elap <- proc.time() - jsttime
      evec <- c(evec,elap["elapsed"])
 
      plot(pvec,avec)

      idone <- idone+1
   }
   iprobdone <- iprobdone+1
}
qplot(pvec,avec) + geom_smooth()

df <- data.frame(prob=pvec,acc=avec,elap=evec)
for (i in 1:20)
{
  vcmd3 <- sprintf("df$v%d <- v%d",i,i)
  eval(parse(text=vcmd3))
}
write.csv(df,"dftest3v-3it.csv")
