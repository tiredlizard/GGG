# install.packages('remotes')
# remotes::install_github('rstudio/tensorflow')
# reticulate::install_python()
# keras::install_keras()

library(tidymodels)
library(tidyverse)
library(vroom)
library(keras)

ggg_train <- vroom('train.csv')
ggg_test <- vroom('test.csv')

nn_recipe <- recipe(type ~ ., data = ggg_train) %>%
  update_role(id, new_role = 'id') %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>% #turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50 #or 100 or 250
                ) %>%
  set_engine('keras') %>% #verbose = 0 prints off less
  set_mode('classification')

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1,10)),
                            levels = 3) ## 10 maxHiddenUnits

nn_folds <- vfold_cv(ggg_train, v = 4, repeats = 1)

tuned_nn_acc <- nn_wf %>%
  tune_grid(resamples = nn_folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

tuned_nn_acc %>% collect_metrics() %>%
  filter(.metric == 'accuracy') %>%
  ggplot(aes(x=hidden_units, y = mean)) + 
  geom_line() +
  theme_minimal()

# bestTune_nn <- tuned_nn %>%
#   select_best(metric = 'roc_auc')
# 
# nn_wf <- 
#   nn_wf %>%
#   finalize_workflow(bestTune_nn) %>%
#   fit(data = ggg_train)
# 
# ## This takes a few min (10 on my laptop) so run it on becker if you want
# 
# nn_preds <- predict(nn_wf, new_data=ggg_test, type = "prob") 
# 
# kag_nn <- nn_preds %>%
#   bind_cols(ggg_test) %>%
#   rename(ACTION=.pred_1) %>%
#   select(id, ACTION)
# 
# vroom_write(kag_nn, "nn.csv", delim = ",")