# STA 521 - Cloud Detection Project 
By: Alison Reynolds and Jackie Du 

### Overview 
This code aims to fit a model that classifies cloudy and non-cloudy areas in satellite images of polar regions. The following sections outline how we conduct exploratory data analysis to identify predictor relationships, fit classification models (i.e. logistic, LDA, QDA, random forest, boosted trees) including tuning its parameters, and finally run diagnostics and an in-depth analysis of the best performing model (in this case, using random forest). 

### Exploratory Data Analysis & Data Splitting
#### EDA 
Our code first creates a table that computes the composition of cloudy and clear pixels for each image as well as in aggregate. It then examines relationships between predictors and with respect to the classification variable. It plots histograms of each predictor segmented by expert label, as well as computes a correlation matrix between all predictors. 

#### Data Splitting
In order to split the data into training, validation, and test sets, our code splits it in two non-trivial ways. The first method is dubbed the 'image split,' in which each image is treated as a complete data set and the three sets are randomly sampled from the list of images. 

The second method is the 'block split' in which we segment each image into equal blocks, and randomly sample blocks from all images to comprise the three different data sets. We wrote a function called `split_blocks_main` to process this data splitting method. 
- This function takes inputs for a dataframe containing image data, number of total blocks, number of blocks allotted for training, and number of blocks allotted for validation. The assumptions that this function makes are that the dataframe contains x-coord and y-coord columns and the number of total blocks are perfect squares. Additionally, it derives the number of test blocks from the difference in total blocks with the sum of training and validation blocks provided by the user.
- The function outputs a named list that contains the training, validation, and test sets as a result of the block splitting. These can be referenced using output$test, output$train, and output$val for each of the data sets, for example. 


### Model Fitting & Tuning 
Our primary models of choice are logistic regression, linear discriminant analysis (LDA), quadratic discriminant analyis (QDA), random forest, and boosted trees. We created a generic function called `CVMaster` that handles all of the model fitting for each of these classification methods. 

#### Inputs 
These are the following inputs that the function takes: 
- _X:_ A data frame of only the predictors 
- _y:_ A vector of the corresponding response labels (factor)
- _k:_ Number of folds (integer)
- _loss:_ A vector of loss functions, ie "accuracy" (character) 
- _estimates:_ A vector of estimates, ie c("roc", "conf_mat", "precision") (character)

#### Outputs 
We added the functionality to return a named list of output to increase the efficiency of the code. The results is a named list which can be accessed by subsetting 
the list by the name of the argument passed, e.g. results$roc accesses the dataframe of roc statistics from the results. ROC estimates returns the sensitivity and specificity for the model, the confusion matrix is a matrix true positive, false positive, etc. values, precision returns the proportion of true positives over all predicted postives. 

The loss function specified will return the average across folds as well as within folds, whereas each of the estimates returns only the average across all folds. For all models that require tuning, the results will return for the results for the model whose tuning parameters with the highest mean accuracy - the exact parameters can be found in results$accuracy, for example. 

#### Model Specifics 
There are two approaches in the model for cross validation and calculating the estimates depending on whether the tidymodels package could be used. For those in which the tidy package can be used (i.e. logistic, rf, boosted trees), we take advantage of the built in capabilities in creating folds and tuning parameters using the parsnip package. Otherwise, we create the folds using the caret package and iteratively loop through each fold to fit the models. 

- **Logistic Regression** - tunes the regularization parameter between 0.0001 and 10
- **Random Forest** - tunes a grid of 25 values of minimal node size (min_n) and number of variables to split at each node (mtry)
- **Boosted Trees** - same as random forest 







### Random Forest 

