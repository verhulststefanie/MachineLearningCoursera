---
title: "Qualitative Activity Recognition"
author: "Stefanie Verhulst"
date: "17 december 2015"
output: 
     html_document:
          fig_caption: true
---
# Introduction

This document describes a qualitative analysis of a certain weight lifting exercise. It was created in the context of a Coursera Peer Assessment in the Practical Machine Learning course of the JHU Data Science specialization. Some of the text found in this document originates from the [assignment page](https://class.coursera.org/predmachlearn-035/human_grading/view/courses/975205/assessments/4/submissions).



## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways: 

* Exactly according to the specification (Class A), 
* Throwing the elbows to the front (Class B), 
* Lifting the dumbbell only halfway (Class C), 
* Lowering the dumbbell only halfway (Class D),
* Throwing the hips to the front (Class E).


More information is available on the [website of the original study](http://groupware.les.inf.puc-rio.br/har). 

## Downloading and Reading the Data

The original source of the data in this project can be found [here](http://groupware.les.inf.puc-rio.br/har).
The training data used in this project are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv), and the test data are available [here](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv).
The following code chunk downloads the data (if not already present in .csv format), then reads this data, and caches this result afterwards. Values matching *"#DIV/0!"* will be read in as missing values to avoid those features being read in as factors.


```r
fileTrain <- "pml-training.csv"
fileTesting <- "pml-testing.csv"
# Check if the data is present, otherwise download it
if (!file.exists(fileTrain) | !file.exists(fileTesting)) {
     download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", fileTrain) 
     download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", fileTesting) 
}
trainFull <- read.csv(fileTrain,na.strings=c("NA","#DIV/0!"))
testing <- read.csv(fileTesting,na.strings=c("NA","#DIV/0!"))
# Show some of the data
trainFull[1:6,1:5]
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
```
## Creating a Validation Set
The dataset comes with a supplied test set, however, the *classe* label for these values is not included. As a result, we will be holding out the supplied test set for now, and partition the supplied training set into a training and validation set. The seed is set to allow for consistent results.


```r
set.seed(54321)
inTrain <- createDataPartition(trainFull$classe, p = .6, list = FALSE)
training <- trainFull[inTrain,]
validation <- trainFull[-inTrain,]
```


## Data Preprocessing: Selecting the Predictors
The original dataset contains 160 columns, some of which are more useful than others. There are many tests we can apply to test for usefulness, however, we must be careful to only look at the training data when doing so. After we have made a selection of useful predictors based on the training data, the same subset must be used for the validation and test data. The following code chunk performs a first selection of predictors, throwing away those predictors that have a near zero variance. There is some [debate](http://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them/) as to whether zero variance predictors should always be removed, but for this project we have chosen to do so. In addition, we will be omitting the first 7 columns:

``X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window`` 

As they contain information a typical model should not be using. After all, the experiment setup could have the subjects execute the exercise classes in a specific order, for example, which is not something we want to include in our model.


```r
predictorsTrain1 <- training[,-(1:7)]
predictorsVal1 <- validation[,-(1:7)]
predictorsTest1 <- testing[,-(1:7)]
nzvTrain <- nearZeroVar(predictorsTrain1,saveMetrics = TRUE)$nzv
predictorsTrain2 <- predictorsTrain1[,nzvTrain==FALSE]
# Use training data metrics to subset test and validation set!
predictorsVal2 <- predictorsVal1[,nzvTrain==FALSE]
predictorsTest2 <- predictorsTest1[,nzvTrain==FALSE] 
```
Next, we will be removing those predictors that have missing values in the training set. Note that this constitutes removing *columns*, not rows, with missing values. There are many thousands of missing values, but most are constituted by a few columns that are nearly fully NA. Thus, it is ok to remove these columns - but not all those rows, as this would be removing useful information. There appear to be no columns with only a few NAs: either they are nearly all NA or there are none. Otherwise, if there were a limited amount of NA values it would be interesting to delete those rows. However, with the given data it seems removing the useless columns is better.


```r
noNAsTrain = sapply(predictorsTrain2,function(x) sum(is.na(x))) == 0
predictorsTrain3 <- predictorsTrain2[,noNAsTrain==TRUE]
predictorsVal3 <- predictorsVal2[,noNAsTrain==TRUE]
predictorsTest3 <- predictorsTest2[,noNAsTrain==TRUE]
```

Finally, we take out the class label:


```r
classTrain <- predictorsTrain3$classe
predictorsTrainFinal <- predictorsTrain3[,names(predictorsTrain3)!="classe"]
classVal <- predictorsVal3$classe
predictorsValFinal <- predictorsVal3[,names(predictorsVal3)!="classe"]
predictorsTestFinal <- predictorsTest3
```

# Analysis: Qualitative Activity Recognition
## Model Building
We will be using the [Random Forests](https://en.wikipedia.org/wiki/Random_forest) algorithm by using the R package *caret*'s function *train()* with *method=rf*. Please note that we will not be using the formula notation, as it is [highly inefficient](http://stackoverflow.com/questions/6449588/r-memory-management-advice-caret-model-matrices-data-frames/6555053#6555053). Before fitting the model, the data is preprocessed by normalization through centering and scaling. In addition, 10-fold cross validation is applied to detect and prevent overfitting, as *caret*'s function *train()* handles this for us. Cross validation is especially necessary for random forests models, as they are sensitive to overfitting on top of their long execution time. The seed is set to ensure consistent results.

```r
set.seed(54321)
model <- train(x = predictorsTrainFinal, y = classTrain, method = "rf",
               na.action = na.omit, preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv"), allowParallel = TRUE)
print(model)
```

```
## Random Forest 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered (52), scaled (52) 
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 10597, 10600, 10598, 10599, 10598, 10599, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9902342  0.9876460  0.002893272  0.003660365
##   27    0.9908287  0.9883971  0.003296359  0.004170776
##   52    0.9843752  0.9802316  0.004369622  0.005528399
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```
To illustrate the model we have now created, we plot the importance of each variable in the final model to give us some insight. Note that this plot uses the [Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity), which is a measure of node impurity. Simply said, it is a measure of how often a randomly chosen element from the set would be incorrectly labeled if it were randomly labeled according to the distribution of labels in the subset. A [higher decrease in Gini](http://stackoverflow.com/questions/736514/r-random-forests-variable-importance) means that a particular predictor variable plays a greater role in partitioning the data into the defined classes.

```r
varImpPlot(model$finalModel,main="Variable Importance Plot",pch=19)
title(xlab="Variable Importance for Data Partitioning\n")
```

![**Figure 1:** Variable importance plot, illustrating the importance of each variable in partitioning the data into the defined classes. Variables to the upper right-side are more important when splitting the data, while the bottom left-side of the plot contains less important variables that are still in the final model.](figure/unnamed-chunk-8-1.png) 

## Model Evaluation
In evaluating the model, we differentiate between **in sample error**, also called resubstitution error, and **out of sample error**, sometimes called generalization error. In sample errors are generally more optimistic than out of sample errors due to overfitting. However, it is important to note that the out-of-bag (OOB) error estimate is an exception to this rule, as it has a [slight pessimistic tendency](http://stats.stackexchange.com/questions/105811/how-would-one-formally-prove-that-the-oob-error-in-random-forest-is-unbiased). This small pessimistic bias is due to the fact that OOB error estimates are calculated on training sets that are (at least) one training case smaller than the full training set. We will now be studying the in sample and out of sample error estimates.

### In Sample Error
To evaluate the final model created by the random forests algorithm, we first print it to examine which metrics are already available.

```r
print(model$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, na.action = ..1,      allowParallel = TRUE) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.76%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3344    4    0    0    0 0.001194743
## B   18 2256    5    0    0 0.010092146
## C    0   12 2034    8    0 0.009737098
## D    0    0   25 1902    3 0.014507772
## E    0    1    6    8 2150 0.006928406
```
We note an out-of-bag (OOB) error estimate of ~ 0.76%, and a good looking confusion matrix. However, the OOB error and the confusion matrix shown above are calculated on the training set. We will now be looking at the performance on the created validation set to evaluate the model on true out of sample data.

### Out of Sample Error
First, we use the model to predict the class labels for the validation set, then, we compare it to their true labels. This is done using the confusion matrix as shown below. Note that when using new data in a model created using *caret*'s *train()* function, we do not need to preprocess this new data ourselves. [*Caret* saves the preprocessing parameters](http://stats.stackexchange.com/questions/46216/pca-and-k-fold-cross-validation-in-caret) and will use them to preprocess the validation data during the predict call, in the same way as the training data was when modeling. 


```r
predictions <- predict(model, predictorsValFinal) 
confM <- confusionMatrix(predictions, classVal)
confM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   18    0    0    0
##          B    3 1494    6    1    1
##          C    0    6 1359   16    3
##          D    0    0    3 1269    3
##          E    0    0    0    0 1435
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9924          
##                  95% CI : (0.9902, 0.9942)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9903          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9842   0.9934   0.9868   0.9951
## Specificity            0.9968   0.9983   0.9961   0.9991   1.0000
## Pos Pred Value         0.9920   0.9927   0.9819   0.9953   1.0000
## Neg Pred Value         0.9995   0.9962   0.9986   0.9974   0.9989
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1904   0.1732   0.1617   0.1829
## Detection Prevalence   0.2864   0.1918   0.1764   0.1625   0.1829
## Balanced Accuracy      0.9977   0.9912   0.9948   0.9929   0.9976
```
Notably, we see an accuracy of 99.2352791% for out of sample data, with a very significant P-value. We note that we did not exceed our OOB error estimate of ~ 0.76%, and have thus successfully avoided overfitting through the cross validation performed by *caret*'s *train()* function. Again, we see a good looking confusion matrix, which we now visualize with colours. 

```r
new.palette=colorRampPalette(c("darkblue","lightblue"),space="rgb") 
plot(levelplot(confM$table,main ="Visual Confusion Matrix",xlab=NULL,ylab=NULL,col.regions=new.palette(100)))
```

![**Figure 2:** Coloured visualization of the confusion matrix. The lighter, larger values are concentrated on the diagonal, while the darker, near-zero values cover everything else, as desired.](figure/unnamed-chunk-11-1.png) 

# Test Set
The following code chunk was used to generate the answers for the provided test set. The instruction for generating the files has been commented out, and the answers are printed instead.


```r
# The testing set part of the assigment
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
predictionsTest <- predict(model, predictorsTestFinal)
answers <- as.character(predictionsTest)
answers
#pml_write_files(answers)
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

# Appendix

## Software Environment
This section describes the software environment in which this assignment was created. The described environment is not the only one in which this report will work, but proper operation is not guaranteed for divergent systems.

* **Computer Architecture:** Intel Core i7 CPU, Intel HD Graphics GPU
* **Operating System:** 64-bit Windows 8
* **Software Toolchain:** R Studio
* **Supporting Software:** R packages: *caret*, *rattle*, *doParallel*, *randomForest* as loaded below

```r
library(caret)
library(rattle)
library(doParallel)
library(randomForest)
```
* **Dependencies:** The data should be stored in the same folder as this file, otherwise it will be downloaded again.

The following R command prints further session info details.

```r
sessionInfo()
```

```
## R version 3.2.3 (2015-12-10)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows >= 8 x64 (build 9200)
## 
## locale:
## [1] LC_COLLATE=Dutch_Belgium.1252  LC_CTYPE=Dutch_Belgium.1252   
## [3] LC_MONETARY=Dutch_Belgium.1252 LC_NUMERIC=C                  
## [5] LC_TIME=Dutch_Belgium.1252    
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods  
## [8] base     
## 
## other attached packages:
##  [1] qdap_2.2.4             RColorBrewer_1.1-2     qdapTools_1.3.1       
##  [4] qdapRegex_0.6.0        qdapDictionaries_1.0.6 randomForest_4.6-12   
##  [7] doParallel_1.0.10      iterators_1.0.8        foreach_1.4.3         
## [10] rattle_4.0.5           caret_6.0-62           ggplot2_2.0.0         
## [13] lattice_0.20-33        knitr_1.11            
## 
## loaded via a namespace (and not attached):
##  [1] splines_3.2.3       gender_0.5.1        gtools_3.5.0       
##  [4] assertthat_0.1      stats4_3.2.3        xlsxjars_0.6.1     
##  [7] yaml_2.1.13         slam_0.1-32         quantreg_5.19      
## [10] chron_2.3-47        digest_0.6.8        minqa_1.2.4        
## [13] colorspace_1.2-6    htmltools_0.2.6     Matrix_1.2-3       
## [16] plyr_1.8.3          tm_0.6-2            XML_3.98-1.3       
## [19] SparseM_1.7         scales_0.3.0        gdata_2.17.0       
## [22] whisker_0.3-2       lme4_1.1-10         MatrixModels_0.4-1 
## [25] openNLP_0.2-5       reports_0.1.4       mgcv_1.8-9         
## [28] car_2.1-1           slidify_0.5         nnet_7.3-11        
## [31] pbkrtest_0.4-4      NLP_0.1-8           magrittr_1.5       
## [34] evaluate_0.8        nlme_3.1-122        MASS_7.3-45        
## [37] class_7.3-14        rsconnect_0.4.1.4   tools_3.2.3        
## [40] data.table_1.9.6    formatR_1.2.1       stringr_1.0.0      
## [43] xlsx_0.5.7          munsell_0.4.2       plotrix_3.6        
## [46] compiler_3.2.3      e1071_1.6-7         grid_3.2.3         
## [49] RCurl_1.95-4.7      nloptr_1.0.4        RGtk2_2.20.31      
## [52] igraph_1.0.1        bitops_1.0-6        rmarkdown_0.8.1    
## [55] venneuler_1.1-0     gtable_0.1.2        codetools_0.2-14   
## [58] DBI_0.3.1           markdown_0.7.7      reshape2_1.4.1     
## [61] R6_2.1.1            gridExtra_2.0.0     dplyr_0.4.3        
## [64] openNLPdata_1.5.3-2 rJava_0.9-7         stringi_1.0-1      
## [67] Rcpp_0.12.2         rpart_4.1-10        wordcloud_2.5
```

## References
* Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
* http://groupware.les.inf.puc-rio.br/har
* http://stackoverflow.com/questions/6449588/r-memory-management-advice-caret-model-matrices-data-frames/6555053#6555053
* http://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them/
* https://en.wikipedia.org/wiki/Random_forest
* http://stackoverflow.com/questions/736514/r-random-forests-variable-importance
* http://stats.stackexchange.com/questions/105811/how-would-one-formally-prove-that-the-oob-error-in-random-forest-is-unbiased
* https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
* http://stats.stackexchange.com/questions/46216/pca-and-k-fold-cross-validation-in-caret
