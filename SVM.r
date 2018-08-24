# SVM Handwritten Digit Recogniser

#####################################################################################
#
# 1. Business Understanding: 
#
# Support Vector Machine which correctly classifies
# the handwritten digits (0-9) based on the pixel values given as features.
#
#####################################################################################

# Loading Libraries
library(dplyr);
library(kernlab);
library(caret);

#Loading data
mnist_train <- read.csv('mnist_train.csv',header=FALSE);
mnist_test <- read.csv('mnist_test.csv',header=FALSE);

# Understanding the data
dim(mnist_train);
str(mnist_train);

names(mnist_train);

# Naming the first column as digit_label
names(mnist_train)[1]<-'digit_label';
names(mnist_test)[1]<-'digit_label';

# Converting digit_label to factor
mnist_train$digit_label<-factor(mnist_train$digit_label);
mnist_test$digit_label<-factor(mnist_test$digit_label);


#Checking for missing values
sapply(mnist_train, function(x) sum(is.na(x)));

sapply(mnist_test, function(x) sum(is.na(x)));
# No missing values.


# Scaling training and testing data
mnist_train[,2:ncol(mnist_train)] <- mnist_train[,2:ncol(mnist_train)]/255;
mnist_test[,2:ncol(mnist_test)] <- mnist_test[,2:ncol(mnist_test)]/255;


#Taking 10% samples from training dataset

set.seed(100)
train.indices = sample(1:nrow(mnist_train), 0.1*nrow(mnist_train))
train_sample = mnist_train[train.indices, ]

#Constructing Model

#Using Linear Kernel
Model_linear <- ksvm(digit_label~ ., data = train_sample, scaled = FALSE, kernel = "vanilladot")
Eval_linear<- predict(Model_linear, mnist_test)

#confusion matrix - Linear Kernel
confusionMatrix(Eval_linear,mnist_test$digit_label)
# Accuracy using linear model is 91.49%

# Model with higher value of C
Model_linear_1 <- ksvm(digit_label~ ., data = train_sample, scaled = FALSE, kernel = "vanilladot", C=10)
Eval_linear_1<- predict(Model_linear_1, mnist_test)

#confusion matrix - Linear Kernel 1
confusionMatrix(Eval_linear_1,mnist_test$digit_label)
# Accuracy using linear model is 91.26%


#using polynomial kernel and default c=1.
Model_linear_2 <- ksvm(digit_label~ ., data = train_sample, scaled = FALSE, kernel = "polydot", C=1)
Eval_linear_2<- predict(Model_linear_2, mnist_test)

#confusion matrix - polynomial Kernel
confusionMatrix(Eval_linear_2,mnist_test$digit_label)
# Accuracy using polynomial kernel is 91.48%

#Using RBF Kernel- Radial kernel
Model_RBF <- ksvm(digit_label~ ., data = train_sample, scaled = FALSE, kernel = "rbfdot")
Eval_RBF<- predict(Model_RBF, mnist_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF,mnist_test$digit_label)
# Accuracy using RBF Kernel is 95.2% and therefore RBF performs better than the linear model.

# Highest Accuracy for RBF Kernel.


# Trying different C and sigma values.

Model_RBF1 <- ksvm(digit_label~ ., data = train_sample, C=3, kpar=list(sigma = 0.01), scaled = FALSE, kernel = "rbfdot")
Eval_RBF1<- predict(Model_RBF1, mnist_test)

#confusion matrix - RBF Kernel
confusionMatrix(Eval_RBF1,mnist_test$digit_label)

# Accuracy using RBF Kernel is 95.73% with C=3 and sigma=0.01






# Choosing RBF kernel for cross-validation to identify optimal C and sigma.



############   Hyperparameter tuning and Cross Validation #####################

# We will use the train function from caret package to perform Cross Validation. 


trainControl <- trainControl(method="cv", number=2)


# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.

set.seed(7)
grid <- expand.grid(.sigma=c(0.1,1,5,10), .C=c(0.1,1,5,10) )


#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(digit_label~., data=train_sample, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)

print(fit.svm)

plot(fit.svm)


# Result
#The best model which gives highest accuracy is via RBF with C=3 and sigma=0.01.
