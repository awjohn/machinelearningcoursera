# Practical Machine Learning - Course Project
=============================================

## Introduction

The goal of this project is to predict the manner in which data collection exercise
that collected a large amount of data about personal activity  [Reference:](http://groupware.les.inf.puc-rio.br/har)
was carried out. 



## Read training data file ...

```r
library(caret)
pmldata <- read.csv("pml-training.csv", na.strings=c("NA", "NULL","Not Available",""))
dim(pmldata)
```

```
## [1] 19622   160
```

```r
pmlTestdata <- read.csv("pml-testing.csv", na.strings=c("NA", "NULL","Not Available",""))
dim(pmlTestdata)
```

```
## [1]  20 160
```

## ... remove cols with no data

```r
isNA <- apply(pmldata, c(2), function(x){sum(is.na(x))})
pmldata <- subset(pmldata[, which(isNA == 0)], 
                    select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

pmlTestdata <- subset(pmlTestdata[, which(isNA == 0)], 
                    select=-c(X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window))

## given the same seed, you get the same sequence.
set.seed(125) 
inTrain <- createDataPartition(y=pmldata$classe, p = 0.75,list=FALSE)
training <- pmldata[ inTrain,]
testing <- pmldata[-inTrain,]

## Training methodology: Random Forest
```

```r
ctrl <- trainControl(method="cv", number=4)
modFit <- train(classe~., data=training, method="rf", ntree=10)
rfPred <- predict(modFit, newdata=testing)
##rfPred
```

## Check predictions against the partitioned test data set.

```r
sum(rfPred == testing$classe)/length(rfPred)
```

```
## [1] 0.9894
```

```r
confusionMatrix(testing$classe, rfPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    0    2    0    0
##          B    3  942    3    1    0
##          C    0    6  846    3    0
##          D    0    1   21  781    1
##          E    0    0    1   10  890
## 
## Overall Statistics
##                                         
##                Accuracy : 0.989         
##                  95% CI : (0.986, 0.992)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.987         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.993    0.969    0.982    0.999
## Specificity             0.999    0.998    0.998    0.994    0.997
## Pos Pred Value          0.999    0.993    0.989    0.971    0.988
## Neg Pred Value          0.999    0.998    0.993    0.997    1.000
## Prevalence              0.285    0.194    0.178    0.162    0.182
## Detection Rate          0.284    0.192    0.173    0.159    0.181
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.995    0.983    0.988    0.998
```


## Check Random Forest variable importance

```r
testRFPred <- predict(modFit, newdata=pmlTestdata)

##testRFPred
##pml_write_files(testRFPred)

varImp(modFit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 52)
## 
##                      Overall
## roll_belt              100.0
## pitch_forearm           64.8
## yaw_belt                63.0
## pitch_belt              54.1
## magnet_dumbbell_y       53.1
## roll_forearm            48.6
## magnet_dumbbell_z       47.8
## accel_dumbbell_y        27.1
## magnet_belt_y           26.0
## roll_dumbbell           24.9
## accel_forearm_x         21.5
## magnet_belt_z           20.2
## magnet_forearm_z        20.1
## magnet_dumbbell_x       20.0
## accel_belt_z            18.3
## yaw_arm                 17.6
## total_accel_dumbbell    16.3
## accel_dumbbell_z        15.7
## accel_forearm_z         15.4
## gyros_belt_z            13.4
```

## Train smaller Random Forest Model

```r
smallpmlData <- subset(pmldata, 
                    select=c(roll_belt, pitch_forearm, yaw_belt, magnet_dumbbell_y, pitch_belt, magnet_dumbbell_z, roll_forearm, accel_dumbbell_y, roll_dumbbell, magnet_dumbbell_x,classe))
smallerModFit <- train(classe ~ ., data=smallpmlData[inTrain,], model="rf", ntree=10)

predict(smallerModFit, newdata=pmlTestdata)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

```r
smallPred <- predict(smallerModFit, newdata=testing)
sum(smallPred == testing$classe) / length(smallPred)
```

```
## [1] 0.981
```

```r
confusionMatrix(testing$classe, smallPred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1383    6    4    1    1
##          B   10  917   15    7    0
##          C    0   11  840    4    0
##          D    0    2   11  786    5
##          E    0    9    0    7  885
## 
## Overall Statistics
##                                         
##                Accuracy : 0.981         
##                  95% CI : (0.977, 0.985)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.976         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.993    0.970    0.966    0.976    0.993
## Specificity             0.997    0.992    0.996    0.996    0.996
## Pos Pred Value          0.991    0.966    0.982    0.978    0.982
## Neg Pred Value          0.997    0.993    0.993    0.995    0.999
## Prevalence              0.284    0.193    0.177    0.164    0.182
## Detection Rate          0.282    0.187    0.171    0.160    0.180
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.995    0.981    0.981    0.986    0.995
```

```r
print(modFit$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 10, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 10
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 3.01%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4091   34    8   10    1     0.01279
## B   67 2683   36   20   19     0.05027
## C    4   50 2442   33    8     0.03745
## D    9   17   55 2287   16     0.04069
## E    4   20    9   18 2629     0.01903
```

## Graphics
Here is a plot of the model's prediction relationship to the testing data

```r
qplot(rfPred, classe, data=testing)
```

![plot of chunk scatterplot](figure/scatterplot.png) 



## Conclusion
- Random forest
  - iterative split into groups
  - Classification tree to explore interation between variables
Cross validation
- used cross validation on the training control with 
- a crossvalidation with exactly the same splitting 
- very slow, took 25 minutes to run
- cross validate to pick predictors estimate requires estimate errors on independent data
- In random forests, there is no need for cross-validation or a separate test 
set to get an unbiased estimate of the test set error. It is estimated internally

what you think the expected out of sample error is
- In sample error < out of sample error guaranteeing that overfitting did not occur
- the out of sample error estimate from the confusion matrix
why you made the choices you did
- Other model were not very accurate and were very time consuming
