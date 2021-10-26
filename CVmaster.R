CVmaster <- function(model=c("lda","qda","rf"),X,y,k,loss=c("class_accur")){
  require(caret)
  folds <- createFolds(y,k)
  
  crossVal <- rep(NA,k)
  for (i in 1:k){
    mod <- train(x=X[-folds[[i]],],y=y[-folds[[i]]],method = model)
    preds <- predict(mod,X[folds[[i]],])
    if (loss == "class_accur"){
      crossVal[i] <- mean(preds != y[folds[[i]]])
    }
  }
  return(mean(crossVal))
}