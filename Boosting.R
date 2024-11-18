library(bonsai)
library(lightgbm)
library(vroom); library(tidymodels); library(tidyverse)
library(dbarts)

ggg_train <- vroom('train.csv')
ggg_test <- vroom('test.csv')

ggg_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>% #turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine('lightgbm') %>% 
  set_mode("classification")

bart_model <- bart(trees = tune()) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode('classification')

boost_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(boost_model)

boost_grid <- grid_regular(trees(),
                           learn_rate(),
                           tree_depth(),
                           levels = 3)

boost_folds <- vfold_cv(ggg_train, v = 5, repeats = 1) # v is number of groups

CV_results <- boost_wf %>%
  tune_grid(resamples = boost_folds, 
            grid = boost_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <-
  boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = ggg_train)

boost_preds <- predict(final_wf, new_data=ggg_test, type = "class")

kag_boost_ggg <- boost_preds %>%
  bind_cols(., ggg_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(kag_boost_ggg, "kag_boost_ggg.csv", delim = ",")
