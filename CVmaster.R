CVmaster <- function(model=c("lda","qda","rf","logistic","svmLinear"),X,y,k,loss=c("accuracy","error"),ntree=NA){
  require(caret)
  require(tidyverse)
  folds <- createFolds(y,k)
  
  # crossVal <- rep(NA,k)
  trying <- tibble()
  for (i in 1:k){
    if (model == "logistic"){
      mod <- train(x=X[-folds[[i]],],y=y[-folds[[i]]],method = "glmnet",family="binomial")
    }
    else if (model == "rf"){
      mod <- train(x=X[-folds[[i]],],y=y[-folds[[i]]],method = "glmnet",family="binomial",ntree=ntree)
    }
    else{
    mod <- train(x=X[-folds[[i]],],y=y[-folds[[i]]],method = model)
    }
    preds <- predict(mod,X[folds[[i]],])
    probs <- predict(mod,X[folds[[i]],], type = 'prob' )
    if (loss == "accuracy"){
      # crossVal[i] <- mean(preds == y[folds[[i]]])
      crossVal <- mean(preds == y[folds[[i]]])
    }
    else if (loss == "error"){
      # crossVal[i] <- mean(preds != y[folds[[i]]])
      crossVal <- mean(preds != y[folds[[i]]])
    }
    
    trying <- bind_rows(trying, data.frame(fold = i, cval = crossVal, probs = probs, ylabels = y[folds[[i]]]))
  }
  # return(crossVal)
  return(trying)
}