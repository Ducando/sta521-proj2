CVmaster <- function(model,X,y,k,loss){
  library(caret)
  folds <- createFolds(y,k)
  
  crossVal <- rep(NA,k)
  for (i in 1:k){
  }
}