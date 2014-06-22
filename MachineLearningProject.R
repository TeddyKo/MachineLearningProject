
## Load required libraries
library(AppliedPredictiveModeling)
library(caret)
library(rattle)
library(rpart.plot)
library(randomForest)

## Import the data sets and treat empty values as NA.
trainingCSV <- read.csv("./pml-training.csv", na.strings=c("NA",""), header=TRUE)
colnames_train <- colnames(trainingCSV)
testingCSV <- read.csv("./pml-testing.csv", na.strings=c("NA",""), header=TRUE)
colnames_test <- colnames(testingCSV)

## Verify that the column names are identical in the training and test set.
## Note: excluding classe in training set and problem_id in testing set
all.equal(colnames_train[1:length(colnames_train)-1], colnames_test[1:length(colnames_train)-1])

## Drop NA columns and other unnecessary columns.
nonNAs <- function(x) {
	as.vector(apply(x, 2, function(x) length(which(!is.na(x)))))
}
## Find missing data (NA) columns to drop.
colcnts <- nonNAs(trainingCSV)
drops <- c()
for (cnt in 1:length(colcnts)) {
    if (colcnts[cnt] < nrow(trainingCSV)) {
        drops <- c(drops, colnames_train[cnt])
    }
}

## Drop NA columns and the first 7 columns
trainingCSV <- trainingCSV[,!(names(trainingCSV) %in% drops)]
trainingCSV <- trainingCSV[,8:length(colnames(trainingCSV))]
testingCSV <- testingCSV[,!(names(testingCSV) %in% drops)]
testingCSV <- testingCSV[,8:length(colnames(testingCSV))]

## Show remaining columns.
# colnames(trainingCSV)
# colnames(testingCSV)

## Check the remained columns to see which have virtually no variability.
# nzv <- nearZeroVar(trainingCSV, saveMetrics=TRUE)
# nzv

## Prepare training, validation and testing data sets
set.seed(1234)
inTrain <- createDataPartition(trainingCSV$classe, p=0.60,list=FALSE)
training <- trainingCSV[inTrain, ]
validation <- trainingCSV[-inTrain, ]
# summary(training)
# head(training)
dim(training)
dim(validation)

## Classification Tree:
## --------------------
set.seed(1234)
## Train on training set with both preprocessing and cross validation.
# model <- train(training$classe ~ ., data=training, method="rpart")
model <- train(training$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = training, method="rpart")
print(model, digits=3)
print(model$finalModel, digits=3)
fancyRpartPlot(model$finalModel)

## Run against validation set.
predictions <- predict(model, newdata=validation)
print(confusionMatrix(predictions, validation$classe), digits=4)
## Run against testing set.
print(predict(model, newdata=testingCSV))
# > print(predict(model, newdata=testingCSV))
#  [1] C A C A A C C A A A C C C A C A A A A C
# Levels: A B C D E
# >

## Random Forest:
## --------------
set.seed(1234)
## Train on training set with both preprocessing and cross validation.
# model <- train(training$classe ~ ., data=training, method="rf")
model <- train(training$classe ~ .,  preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = training, method="rf")
print(model, digits=3)

## Run against validation set.
predictions <- predict(model, newdata=validation)
print(confusionMatrix(predictions, validation$classe), digits=4)
## Run against testing set.
print(predict(model, newdata=testingCSV))
# > print(predict(model, newdata=testingCSV))
#  [1] B A B A A E D B A A B C B A E E A B B B
# Levels: A B C D E
# >

## Out of Sample Error: the error rate we get on new data set.
## -----------------------------------------------------------
# Classification Tree: on validation data set = 1 - 0.4904
# Random Forest:       on validation data set = 1 - 0.9898
#
## The accuracy of Random Forest model looks fine; 
## but the accuracy of Classification Tree still needs some work.


