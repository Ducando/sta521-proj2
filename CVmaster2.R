CVmaster2 <- function (model,X,y,k,loss, ntree=NA) {
  require(tidyverse)
  require(tidymodels)
  require(workflows)
  require(tune)
  require(tibble)
  # require(caret)
  require(rsample)
  require(xgboost)
  # require(parallel)
  
  # check if model in one of the options 
  model_options <- c("lda","qda","rf","logistic","svm", "rf_boost")
  if(!(model %in% model_options)){
    message(paste0("Model not available, please choose from: lda, qda, rf, logistic, svm, rf_boost"))
    break
  }
  
  # split data into folds 
  data <- bind_cols(X, labels = y)
  set.seed(513)
  folds <- vfold_cv(data, v = k)
  
  # initializing final returned objects 
  estimates <- list()
  
  # fit models 
  if(model == "logistic"){
    # specify model
    mod <- logistic_reg(penalty = tune(), mixture =1) %>%
      set_engine("glmnet")
    
    # make recipe, normalize predictors 
    recipe <- 
      recipe(labels ~., data = data) %>%
      step_normalize(all_predictors())
    
    # make workflow 
    workflow <- workflow() %>%
      add_model(mod) %>%
      add_recipe(recipe)
    
    # create tuning grid 
    reg_grid <- tibble(penalty = 10^(seq(-4, 1, length.out = 30)))
    
    # fit model 
    res <- workflow %>%
      tune_grid(grid = reg_grid,
                control = control_grid(save_pred = TRUE),
                resamples = folds)
    
    return(res)
    
    
    ## FIND ACCURACIES
    # find best parameter based on accuracy 
    # if loss contains accuracy
    if ("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best
      
      # get mean accuracy based on best paramater for each fold
      res %>%
        select(id, .metrics) %>%
        hoist(.metrics, penalty = "penalty", metric = ".metric", estimate = ".estimate") %>%
        select(-.metrics) %>%
        unnest_longer(penalty) %>%
        filter(penalty == best$penalty) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        select(id, penalty, metric, mean) %>%
        distinct() %>%
        mutate(model = "Logistic Regression") -> accuracies
      
      # append to estimates
      estimates <- c(estimates, accuracy = list(accuracies))
    } # end accuracy 
    if ("roc" %in% loss){
      ## FIND ROC VALUES
      # if roc in loss
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Logistic Regression") -> roc
      
      # append to estimates 
      estimates <- c(estimates, roc = list(roc))
    } # end roc 
    
    
    
  } # end logistic 
  else if (model == "rf") {
    # cores <- parallel::detectCores()
    
    # specify model 
    mod <- rand_forest(mtry = tune(), min_n = tune(), trees = 5) %>%
      set_engine("ranger") %>%
      set_mode("classification")
    
    workflow <- workflow() %>%
      add_model(mod) %>%
      add_formula(labels ~.)
    
    ## CHANGE TO 25 TO RUN AT END
    res <- workflow %>%
      tune_grid(grid = 5, 
                control = control_grid(save_pred = TRUE),
                resamples = folds)
    
    # if accuracy
    if("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best
      
      res %>%
        select(id, .metrics) %>%
        hoist(.metrics, mtry = "mtry", min_n = "min_n", metric = ".metric", estimate = ".estimate") %>%
        select(-.metrics) %>%
        unnest_longer(min_n) %>%
        filter(min_n == best$min_n) %>%
        unnest_longer(mtry) %>%
        filter(mtry == best$mtry) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        select(id, mtry, min_n, metric, mean) %>%
        distinct() %>%
        mutate(model = "Random Forest") -> accuracies
      
      # append to estimates
      estimates <- c(estimates, accuracy = list(accuracies))
    } # end accuracies 
    
    # if roc
    if("roc" %in% loss){
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Random Forest") -> roc
      
      # append to estimates 
      estimates <- c(estimates, roc = list(roc))
      
    } # end roc 
    
    
  } # end random forest 
  else if (model == "lda") {


    
  } # end lda 
  else if (model == "qda") {
    
  } # end qda 
  else if (model == "rf_boost"){
    
    # add more arguments?
    mod <- boost_tree(mtry = tune(), min_n = tune(), trees = 5) %>% 
      set_mode("classification") %>% 
      set_engine("xgboost")
    
    workflow <- workflow() %>%
      add_model(mod) %>%
      add_formula(labels ~.)
    
    ## CHANGE TO 25 TO RUN AT END
    res <- workflow %>%
      tune_grid(grid = 5, 
                control = control_grid(save_pred = TRUE),
                resamples = folds)
    
    # if accuracy
    if("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best

      res %>%
        select(id, .metrics) %>%
        hoist(.metrics, mtry = "mtry", min_n = "min_n", metric = ".metric", estimate = ".estimate") %>%
        select(-.metrics) %>%
        unnest_longer(min_n) %>%
        filter(min_n == best$min_n) %>%
        unnest_longer(mtry) %>%
        filter(mtry == best$mtry) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        select(id, mtry, min_n, metric, mean) %>%
        distinct() %>%
        mutate(model = "Boosted Trees") -> accuracies

      # append to estimates
      estimates <- c(estimates, accuracy = list(accuracies))
    } # end accuracies

    # if roc
    if("roc" %in% loss){
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Boosted Trees") -> roc

      # append to estimates
      estimates <- c(estimates, roc = list(roc))

    } # end roc
    
    
    
    
    
    
    
  } # end boost 
  else {
    
  } # end svm 
  
  # return(estimates)
}







