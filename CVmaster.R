CVmaster <- function(model=c("lda","qda","rf","logistic","svmLinear"),X,y,k,loss=c("accuracy","error"),ntree=NA){
  require(caret)
  folds <- createFolds(y,k)
  
  crossVal <- rep(NA,k)
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
    if (loss == "accuracy"){
      crossVal[i] <- mean(preds == y[folds[[i]]])
    }
    else if (loss == "error"){
      crossVal[i] <- mean(preds != y[folds[[i]]])
    }
  }
  return(crossVal)
}