Practical Machine Learning Course Project 
========================================================

The goal of this project is to predict the manner in which they did the exercise. Due to the limit of the available time and new to the RStudio, I have decided to test RandomForest algorithm first, and will add and compare with other algorithms later. 

Load required libraries and set seed for reproducibility


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
set.seed(1234)
```

Read data files and make training and validation data sets.
(1) Assume the data files are in current working directory.
(2) 60% for training and 40% for validation - since the data is big enough.


```r
rm(list=ls())
fileTraining <- "C:/Users/TeddyKo/R_working/pml-training.csv"
trainingCSV = read.csv(fileTraining,na.strings=c("NA",""), header=TRUE)
inTrain <- createDataPartition(trainingCSV$classe, p=0.60,list=FALSE)
training <- trainingCSV[inTrain, ]
validation <- trainingCSV[-inTrain, ]
```

Create summary and view the top few records


```r
## summary(training)
## head(training)
```

Some of the columns have no data, it might not be productive to use them as-is. Therefore filtering out fields with a lot of (more than 60%) null values.


```r
goodVars <- c((colSums(is.na(training[,-160])) >= 0.4*nrow(training)),160)
training <- training[,goodVars]
dim(training)
```

```
## [1] 11776   101
```

```r
validation<-validation[,goodVars]
dim(validation)
```

```
## [1] 7846  101
```

```r
training<-training[complete.cases(training),]
dim(training)
```

```
## [1] 11776   101
```

Train the model using RandomForest on the training data set.


```r
model <- randomForest(classe ~ ., data=training)
print(model)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 10
## 
##         OOB estimate of  error rate: 0.01%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3348    0    0    0    0   0.0000000
## B    1 2278    0    0    0   0.0004388
## C    0    0 2054    0    0   0.0000000
## D    0    0    0 1930    0   0.0000000
## E    0    0    0    0 2165   0.0000000
```

Evaluate the model on the evaluation dataset.


```r
confusionMatrix(predict(model,newdata=validation[,-ncol(validation)]), validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    1    0    0    0
##          B    0 1517    1    0    0
##          C    0    0 1367    1    0
##          D    0    0    0 1285    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                     
##                Accuracy : 0.999     
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 0.999     
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.999    0.999    0.999    0.999
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    0.999    0.999    0.999    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.193    0.174    0.164    0.184
## Detection Prevalence    0.285    0.193    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
accurate <- c(as.numeric(predict(model,newdata=validation[,-ncol(validation)])==validation$classe))
accuracy <- sum(accurate)*100/nrow(validation)
message("Model Accuracy using Validation set = " , format(round(accuracy, 2), nsmall = 2), "%")
```

```
## Model Accuracy using Validation set = 99.95%
```

Predict the class values in the testing set with 20 observations


```r
fileTesting <- "C:/Users/TeddyKo/R_working/pml-testing.csv"
testing = read.csv(fileTesting, na.strings=c("NA",""), header=TRUE)
dim(testing)
```

```
## [1]  20 160
```

```r
testing <- testing[,goodVars]
dim(testing)
```

```
## [1]  20 101
```

```r
predictions <- predict(model,newdata=testing)
predictions
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A  A 
## Levels: A B C D E
```

