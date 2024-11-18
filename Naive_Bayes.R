library(tidyverse)
library(tidymodels)
library(vroom)
library(discrim)

ggg_train <- vroom('train.csv')
ggg_test <- vroom('test.csv')

ggg_recipe <- recipe(type ~ ., data = ggg_train) %>%
  step_mutate(color = as.factor(color)) %>%
  step_dummy(all_nominal_predictors()) %>% #turn color to factor then dummy encode color
  step_normalize(all_numeric_predictors(), min=0, max=1)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_wf <- workflow() %>%
  add_recipe(ggg_recipe) %>%
  add_model(nb_model)

nb_grid <- grid_regular(Laplace(), 
                        smoothness(),
                        levels = 10)

nb_folds <- vfold_cv(ggg_train, v = 15, repeats = 1) # v is number of groups

CV_results <- nb_wf %>%
  tune_grid(resamples = nb_folds, 
            grid = nb_grid,
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <-
  nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = ggg_train)

nb_preds <- predict(final_wf, new_data=ggg_test, type = "class")

nb_preds
# kaggle
kag_nb_ggg <- nb_preds %>%
  bind_cols(., ggg_test) %>%
  rename(type=.pred_class) %>%
  select(id, type)

vroom_write(kag_nb_ggg, "kag_nb_ggg.csv", delim = ",")
