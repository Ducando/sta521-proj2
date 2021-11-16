CVmaster <- function (model,X,y,k,loss = "accuracy", estimates, ntree=NA) {
  require(tidyverse)
  require(tidymodels)
  require(workflows)
  require(tune)
  require(tibble)
  require(rsample)
  require(xgboost)
  require(MASS)
  
  # check if model in one of the options 
  model_options <- c("lda","qda","rf","logistic", "boosted_trees")
  if(!(model %in% model_options)){
    message("Model not available, please choose from: lda, qda, rf, logistic, svm, rf_boost")
    break
  }
  
  # check if loss/estimates is in the options 
  loss_functions <- c("accuracy")
  if(!(loss %in% loss_functions)){
    message("Model not available, please choose from: accuracy")
    break
  }
  
  # split data into folds 
  data <- bind_cols(X, labels = y)
  set.seed(513)
  folds <- vfold_cv(data, v = k)
  
  # initializing final returned objects 
  final_results <- list()
  
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
    
  
    # if loss contains accuracy
    if ("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best
      
      # get mean accuracy based on best paramater for each fold
      res %>%
        dplyr::select(id, .metrics) %>%
        hoist(.metrics, penalty = "penalty", metric = ".metric", estimate = ".estimate") %>%
        dplyr::select(-.metrics) %>%
        unnest_longer(penalty) %>%
        filter(penalty == best$penalty) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        dplyr::select(id, penalty, metric, mean) %>%
        distinct() %>%
        mutate(model = "Logistic Regression") -> accuracies
      
      # append to final_results
      final_results <- c(final_results, accuracy = list(accuracies))
    } # end accuracy 
    # if roc in loss
    if ("roc" %in% estimates){
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Logistic Regression") -> roc
      
      # append to final_results 
      final_results <- c(final_results, roc = list(roc))
    } # end roc 
    
    # if conf_mat
    if("conf_mat" %in% estimates){
      res %>%
        collect_predictions(parameters = best) -> best_res
      
      best_res %>%
        group_by(id) %>%
        conf_mat(labels, `.pred_class`) %>%
        mutate(tidied = map(conf_mat, tidy)) %>%
        unnest(tidied) -> cells_per_resample
      
      counts_per_resample <- best_res %>%
        group_by(id) %>%
        summarize(total = n()) %>%
        left_join(cells_per_resample, by = "id") %>%
        mutate(prop = value/total) %>%
        group_by(name) %>%
        summarize(prop = mean(prop))
      
      mean_cmat <- matrix(counts_per_resample$prop, byrow = TRUE, ncol = 2)
      mean_cmat <- data.frame(mean_cmat)
      rownames(mean_cmat) <- paste0("predict_",levels(best_res$labels))
      colnames(mean_cmat) <- paste0("truth_", levels(best_res$labels))
      
      # append to final_results 
      final_results <- c(final_results, conf_mat = list(mean_cmat))
      
    } # end conf_mat
    
    # if precision
    if("precision" %in% estimates){
      # TP / (TP + FP)
      precision <- mean_cmat["predict_1", "truth_1"] / 
        (mean_cmat["predict_1", "truth_1"] + mean_cmat["predict_1", "truth_-1"])
      
      # append to final_results 
      final_results <- c(final_results, precision = list(precision))
    } # end precision
    
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
    
    res <- workflow %>%
      tune_grid(grid = 25, 
                control = control_grid(save_pred = TRUE),
                resamples = folds)
    
    # if accuracy
    if("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best
      
      res %>%
        dplyr::select(id, .metrics) %>%
        hoist(.metrics, mtry = "mtry", min_n = "min_n", metric = ".metric", estimate = ".estimate") %>%
        dplyr::select(-.metrics) %>%
        unnest_longer(min_n) %>%
        filter(min_n == best$min_n) %>%
        unnest_longer(mtry) %>%
        filter(mtry == best$mtry) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        dplyr::select(id, mtry, min_n, metric, mean) %>%
        distinct() %>%
        mutate(model = "Random Forest") -> accuracies
      
      # append to final_results
      final_results <- c(final_results, accuracy = list(accuracies))
    } # end accuracies 
    
    # if roc
    if("roc" %in% estimates){
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Random Forest") -> roc
      
      # append to final_results 
      final_results <- c(final_results, roc = list(roc))
      
    } # end roc 
    
    # if conf_mat
    if("conf_mat" %in% estimates){
      res %>%
        collect_predictions(parameters = best) -> best_res
      
      best_res %>%
        group_by(id) %>%
        conf_mat(labels, `.pred_class`) %>%
        mutate(tidied = map(conf_mat, tidy)) %>%
        unnest(tidied) -> cells_per_resample
      
      counts_per_resample <- best_res %>%
        group_by(id) %>%
        summarize(total = n()) %>%
        left_join(cells_per_resample, by = "id") %>%
        mutate(prop = value/total) %>%
        group_by(name) %>%
        summarize(prop = mean(prop))
      
      mean_cmat <- matrix(counts_per_resample$prop, byrow = TRUE, ncol = 2)
      mean_cmat <- data.frame(mean_cmat)
      rownames(mean_cmat) <- paste0("predict_",levels(best_res$labels))
      colnames(mean_cmat) <- paste0("truth_", levels(best_res$labels))
      
      # append to final_results 
      final_results <- c(final_results, conf_mat = list(mean_cmat))
      
    } # end conf_mat
    
    # if precision
    if("precision" %in% estimates){
      # TP / (TP + FP)
      precision <- mean_cmat["predict_1", "truth_1"] / 
        (mean_cmat["predict_1", "truth_1"] + mean_cmat["predict_1", "truth_-1"])
      
      # append to final_results 
      final_results <- c(final_results, precision = list(precision))
    } # end precision
    
    
  } # end random forest 
  else if (model == "boosted_trees"){

    mod <- boost_tree(mtry = tune(), min_n = tune(), trees = 5) %>% 
      set_mode("classification") %>% 
      set_engine("xgboost")
    
    workflow <- workflow() %>%
      add_model(mod) %>%
      add_formula(labels ~.)
    
    res <- workflow %>%
      tune_grid(grid = 25, 
                control = control_grid(save_pred = TRUE),
                resamples = folds)
    
    # if accuracy
    if("accuracy" %in% loss){
      res %>% select_best("accuracy")  -> best
      
      res %>%
        dplyr::select(id, .metrics) %>%
        hoist(.metrics, mtry = "mtry", min_n = "min_n", metric = ".metric", estimate = ".estimate") %>%
        dplyr::select(-.metrics) %>%
        unnest_longer(min_n) %>%
        filter(min_n == best$min_n) %>%
        unnest_longer(mtry) %>%
        filter(mtry == best$mtry) %>%
        unnest_longer(metric) %>%
        filter(metric == "accuracy") %>%
        unnest_longer(estimate) %>%
        group_by(id) %>%
        mutate(mean = mean(estimate)) %>%
        dplyr::select(id, mtry, min_n, metric, mean) %>%
        distinct() %>%
        mutate(model = "Boosted Trees") -> accuracies
      
      # append to final_results
      final_results <- c(final_results, accuracy = list(accuracies))
    } # end accuracies
    
    # if roc
    if("roc" %in% estimates){
      res %>%
        collect_predictions(parameters = best) %>%
        roc_curve(labels, `.pred_-1`) %>%
        mutate(model = "Boosted Trees") -> roc
      
      # append to final_results
      final_results <- c(final_results, roc = list(roc))
      
    } # end roc
    
    # if conf_mat
    if("conf_mat" %in% estimates){
      res %>%
        collect_predictions(parameters = best) -> best_res
      
      best_res %>%
        group_by(id) %>%
        conf_mat(labels, `.pred_class`) %>%
        mutate(tidied = map(conf_mat, tidy)) %>%
        unnest(tidied) -> cells_per_resample
      
      counts_per_resample <- best_res %>%
        group_by(id) %>%
        summarize(total = n()) %>%
        left_join(cells_per_resample, by = "id") %>%
        mutate(prop = value/total) %>%
        group_by(name) %>%
        summarize(prop = mean(prop))
      
      mean_cmat <- matrix(counts_per_resample$prop, byrow = TRUE, ncol = 2)
      mean_cmat <- data.frame(mean_cmat)
      rownames(mean_cmat) <- paste0("predict_",levels(best_res$labels))
      colnames(mean_cmat) <- paste0("truth_", levels(best_res$labels))
      
      # append to final_results 
      final_results <- c(final_results, conf_mat = list(mean_cmat))
      
    } # end conf_mat
    
    # if precision
    if("precision" %in% estimates){
      # TP / (TP + FP)
      precision <- mean_cmat["predict_1", "truth_1"] / 
        (mean_cmat["predict_1", "truth_1"] + mean_cmat["predict_1", "truth_-1"])
      
      # append to final_results 
      final_results <- c(final_results, precision = list(precision))
    } # end precision
    
    
  } # end boost 
  else if (model == "lda") {
    folds <- caret::createFolds(y,k)
    
    # fit model for each fold, get predictions and accuracy 
    fitModel <- function(i) {
      trainx <- X[-folds[[i]],]
      trainy <- y[-folds[[i]]]
      valx <- X[folds[[i]],]
      valy <- y[folds[[i]]]
      
      train_data <- bind_cols(y = trainy, trainx)
      res <- lda(y ~ ., data = train_data)
      
      preds <- predict(res, valx)
      data.frame(folds = i, preds) %>%
        bind_cols(truth = valy) -> preds
      
      #accuracy per fold
      preds %>%
        accuracy(truth, class) -> accuracy
      
      return(list(preds = preds, accuracy = accuracy$.estimate))
    }
    
    # call function for each fold 
    folds_res <- map(seq(k),
                     function(i) {
                       fitModel(i)
                     })
    
    # aggregate accuracies 
    if("accuracy" %in% loss){
      accuracies <- map(seq(k), function(i){
        folds_res[[i]]$accuracy
      })
      
      accuracies <- data.frame(id = paste0("Fold", str_pad(seq(k), 2, pad="0")), mean = unlist(accuracies), model = "LDA")
      # append to final_results 
      final_results <- c(final_results, accuracy = list(accuracies))
    } # end accuracy 
    
    # combine predictions 
    preds <- map(seq(k), function(i){
      folds_res[[i]]$preds
    })
    preds <- reduce(preds, bind_rows) 
    
    if("roc" %in% estimates){
      # calc roc
      preds %>%
        roc_curve(truth, `posterior..1`) %>%
        mutate(model = "LDA") -> roc
      
      # append to final_results 
      final_results <- c(final_results, roc = list(roc))
    } # end roc
    
    if("conf_mat" %in% estimates){
      # conf matrix 
      preds %>%
        group_by(folds) %>%
        conf_mat(truth, class) %>%
        mutate(tidied = map(conf_mat, tidy)) %>%
        unnest(tidied) -> cells_per_resample
      
      counts_per_resample <- preds %>%
        group_by(folds) %>%
        summarize(total = n()) %>%
        left_join(cells_per_resample, by = "folds") %>%
        mutate(prop = value/total) %>%
        group_by(name) %>%
        summarize(prop = mean(prop))
      
      mean_cmat <- matrix(counts_per_resample$prop, byrow = TRUE, ncol = 2)
      mean_cmat <- data.frame(mean_cmat)
      rownames(mean_cmat) <- paste0("predict_",levels(preds$truth))
      colnames(mean_cmat) <- paste0("truth_", levels(preds$truth))
      
      # append to final_results 
      final_results <- c(final_results, conf_mat = list(mean_cmat))
    } # end conf mat
  
    # if precision
    if("precision" %in% estimates){
      precision <- mean_cmat["predict_1", "truth_1"] / 
        (mean_cmat["predict_1", "truth_1"] + mean_cmat["predict_1", "truth_-1"])
      
      # append to final_results 
      final_results <- c(final_results, precision = list(precision))
    } # end precision
    
  } # end lda 
  else {
    folds <- caret::createFolds(y,k)
    
    # fit model for each fold, get predictions and accuracy 
    fitModel <- function(i) {
      trainx <- X[-folds[[i]],]
      trainy <- y[-folds[[i]]]
      valx <- X[folds[[i]],]
      valy <- y[folds[[i]]]
      
      train_data <- bind_cols(y = trainy, trainx)
      res <- qda(y ~ ., data = train_data)
      
      preds <- predict(res, valx)
      data.frame(folds = i, preds) %>%
        bind_cols(truth = valy) -> preds
      
      #accuracy per fold
      preds %>%
        accuracy(truth, class) -> accuracy
      
      return(list(preds = preds, accuracy = accuracy$.estimate))
    }
    
    # call function for each fold 
    folds_res <- map(seq(k),
                     function(i) {
                       fitModel(i)
                     })
    
    # aggregate accuracies 
    if("accuracy" %in% loss){
      accuracies <- map(seq(k), function(i){
        folds_res[[i]]$accuracy
      })
      
      accuracies <- data.frame(id = paste0("Fold", str_pad(seq(k), 2, pad="0")), mean = unlist(accuracies), model = "QDA")
      # append to final_results 
      final_results <- c(final_results, accuracy = list(accuracies))
    } # end accuracy 
    
    # combine predictions 
    preds <- map(seq(k), function(i){
      folds_res[[i]]$preds
    })
    preds <- reduce(preds, bind_rows) 
    
    if("roc" %in% estimates){
      # calc roc
      preds %>%
        roc_curve(truth, `posterior..1`) %>%
        mutate(model = "QDA") -> roc
      
      # append to final_results 
      final_results <- c(final_results, roc = list(roc))
    } # end roc
    
    if("conf_mat" %in% estimates){
      # conf matrix 
      preds %>%
        group_by(folds) %>%
        conf_mat(truth, class) %>%
        mutate(tidied = map(conf_mat, tidy)) %>%
        unnest(tidied) -> cells_per_resample
      
      counts_per_resample <- preds %>%
        group_by(folds) %>%
        summarize(total = n()) %>%
        left_join(cells_per_resample, by = "folds") %>%
        mutate(prop = value/total) %>%
        group_by(name) %>%
        summarize(prop = mean(prop))
      
      mean_cmat <- matrix(counts_per_resample$prop, byrow = TRUE, ncol = 2)
      mean_cmat <- data.frame(mean_cmat)
      rownames(mean_cmat) <- paste0("predict_",levels(preds$truth))
      colnames(mean_cmat) <- paste0("truth_", levels(preds$truth))
      
      # append to final_results 
      final_results <- c(final_results, conf_mat = list(mean_cmat))
    } # end conf mat
    
    # if precision
    if("precision" %in% estimates){
      precision <- mean_cmat["predict_1", "truth_1"] / 
        (mean_cmat["predict_1", "truth_1"] + mean_cmat["predict_1", "truth_-1"])
      
      # append to final_results 
      final_results <- c(final_results, precision = list(precision))
    } # end precision
    
    
  } # end qda 
  
  return(final_results)
}







