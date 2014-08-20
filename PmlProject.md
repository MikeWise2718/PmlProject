# Judging Quality of Excersice Execution from Accelerator data


Practical Machine Learning - Course Project Writup

Mike Wise - 20 Aug 2014 - predmachlearn-004

This looks at the practicality of juding he quality of excersize execution (weight lifting) from accerlator data gathered from sensors located on 
the subjects body.


# Data Processing
## Loading and preprocessing the data
Here we link in some libraries will will be using in the following code, we presume the directory
has already been set to contain the data.
We then load the data (as a data.table as opposed to a data.frame 
for faster and more flexible manipluation)
## Load the data
Here we load the data 

```r
library(data.table)
library(ElemStatLearn)
library(randomForest)
library(caret)

# Data Processing
## Load the data

otrn <- data.table(read.csv("pml-training.csv"))
```

```
## Warning: cannot open file 'pml-training.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r
otst <- data.table(read.csv("pml-testing.csv"))
```

```
## Warning: cannot open file 'pml-testing.csv': No such file or directory
```

```
## Error: cannot open the connection
```

There are 160 columns in the original data, many of them blank and filled with NA values. We reduce the dataset, throwing
away all columns that are not numeric or integer. We also throw away the column "num_window" as that is not sensor data.


```r
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
```

```
## [1] "6 tst kurtosis_roll_belt is not numeric/integer but logical"
## [1] "7 tst kurtosis_picth_belt is not numeric/integer but logical"
## [1] "8 tst kurtosis_yaw_belt is not numeric/integer but logical"
## [1] "9 tst skewness_roll_belt is not numeric/integer but logical"
## [1] "10 tst skewness_roll_belt.1 is not numeric/integer but logical"
## [1] "11 tst skewness_yaw_belt is not numeric/integer but logical"
## [1] "12 tst max_roll_belt is not numeric/integer but logical"
## [1] "13 tst max_picth_belt is not numeric/integer but logical"
## [1] "14 tst max_yaw_belt is not numeric/integer but logical"
## [1] "15 tst min_roll_belt is not numeric/integer but logical"
## [1] "16 tst min_pitch_belt is not numeric/integer but logical"
## [1] "17 tst min_yaw_belt is not numeric/integer but logical"
## [1] "18 tst amplitude_roll_belt is not numeric/integer but logical"
## [1] "19 tst amplitude_pitch_belt is not numeric/integer but logical"
## [1] "20 tst amplitude_yaw_belt is not numeric/integer but logical"
## [1] "21 tst var_total_accel_belt is not numeric/integer but logical"
## [1] "22 tst avg_roll_belt is not numeric/integer but logical"
## [1] "23 tst stddev_roll_belt is not numeric/integer but logical"
## [1] "24 tst var_roll_belt is not numeric/integer but logical"
## [1] "25 tst avg_pitch_belt is not numeric/integer but logical"
## [1] "26 tst stddev_pitch_belt is not numeric/integer but logical"
## [1] "27 tst var_pitch_belt is not numeric/integer but logical"
## [1] "28 tst avg_yaw_belt is not numeric/integer but logical"
## [1] "29 tst stddev_yaw_belt is not numeric/integer but logical"
## [1] "30 tst var_yaw_belt is not numeric/integer but logical"
## [1] "44 tst var_accel_arm is not numeric/integer but logical"
## [1] "45 tst avg_roll_arm is not numeric/integer but logical"
## [1] "46 tst stddev_roll_arm is not numeric/integer but logical"
## [1] "47 tst var_roll_arm is not numeric/integer but logical"
## [1] "48 tst avg_pitch_arm is not numeric/integer but logical"
## [1] "49 tst stddev_pitch_arm is not numeric/integer but logical"
## [1] "50 tst var_pitch_arm is not numeric/integer but logical"
## [1] "51 tst avg_yaw_arm is not numeric/integer but logical"
## [1] "52 tst stddev_yaw_arm is not numeric/integer but logical"
## [1] "53 tst var_yaw_arm is not numeric/integer but logical"
```

Here we check the quality of the data, showing what columns we have retained (all relevant sensor data columns) and 
showing that we have reduced 


```r
# Check the quality
summary(ntrn)
```

```
##     user_name    classe     roll_belt       pitch_belt    
##  adelmo  :3892   A:5580   Min.   :-28.9   Min.   :-55.80  
##  carlitos:3112   B:3797   1st Qu.:  1.1   1st Qu.:  1.76  
##  charles :3536   C:3422   Median :113.0   Median :  5.28  
##  eurico  :3070   D:3216   Mean   : 64.4   Mean   :  0.31  
##  jeremy  :3402   E:3607   3rd Qu.:123.0   3rd Qu.: 14.90  
##  pedro   :2610            Max.   :162.0   Max.   : 60.30  
##     yaw_belt      total_accel_belt  gyros_belt_x      gyros_belt_y    
##  Min.   :-180.0   Min.   : 0.0     Min.   :-1.0400   Min.   :-0.6400  
##  1st Qu.: -88.3   1st Qu.: 3.0     1st Qu.:-0.0300   1st Qu.: 0.0000  
##  Median : -13.0   Median :17.0     Median : 0.0300   Median : 0.0200  
##  Mean   : -11.2   Mean   :11.3     Mean   :-0.0056   Mean   : 0.0396  
##  3rd Qu.:  12.9   3rd Qu.:18.0     3rd Qu.: 0.1100   3rd Qu.: 0.1100  
##  Max.   : 179.0   Max.   :29.0     Max.   : 2.2200   Max.   : 0.6400  
##   gyros_belt_z     accel_belt_x      accel_belt_y    accel_belt_z   
##  Min.   :-1.460   Min.   :-120.00   Min.   :-69.0   Min.   :-275.0  
##  1st Qu.:-0.200   1st Qu.: -21.00   1st Qu.:  3.0   1st Qu.:-162.0  
##  Median :-0.100   Median : -15.00   Median : 35.0   Median :-152.0  
##  Mean   :-0.130   Mean   :  -5.59   Mean   : 30.1   Mean   : -72.6  
##  3rd Qu.:-0.020   3rd Qu.:  -5.00   3rd Qu.: 61.0   3rd Qu.:  27.0  
##  Max.   : 1.620   Max.   :  85.00   Max.   :164.0   Max.   : 105.0  
##  magnet_belt_x   magnet_belt_y magnet_belt_z     roll_arm     
##  Min.   :-52.0   Min.   :354   Min.   :-623   Min.   :-180.0  
##  1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375   1st Qu.: -31.8  
##  Median : 35.0   Median :601   Median :-320   Median :   0.0  
##  Mean   : 55.6   Mean   :594   Mean   :-346   Mean   :  17.8  
##  3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306   3rd Qu.:  77.3  
##  Max.   :485.0   Max.   :673   Max.   : 293   Max.   : 180.0  
##    pitch_arm         yaw_arm        total_accel_arm
##  Min.   :-88.80   Min.   :-180.00   Min.   : 1.0   
##  1st Qu.:-25.90   1st Qu.: -43.10   1st Qu.:17.0   
##  Median :  0.00   Median :   0.00   Median :27.0   
##  Mean   : -4.61   Mean   :  -0.62   Mean   :25.5   
##  3rd Qu.: 11.20   3rd Qu.:  45.88   3rd Qu.:33.0   
##  Max.   : 88.50   Max.   : 180.00   Max.   :66.0
```

```r
ona <- sum(is.na(otrn))
nna <- sum(is.na(ntrn))
msg <- sprintf("Original training na count:%d  - After processing:%d",ona,nna)
print(msg)
```

```
## [1] "Original training na count:1287472  - After processing:0"
```

# Model Fitting

We fit a models to the data using Random Forests. The accuracy was 100 percent.

# Random Forests

```r
## Random Forests
set.seed(2718)
rffit <- randomForest(classe ~ ., ntrn, importance=T)

prftrn <- predict(rffit, ntrn)
confusionMatrix(prftrn, ntrn$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
varImpPlot(rffit)
```

![plot of chunk unnamed-chunk-4](figure/unnamed-chunk-4.png) 

```r
prftst <- predict(rffit, ntst)
prftst
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
