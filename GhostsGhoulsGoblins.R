library(tidymodels)
library(vroom)

ggg_train <- vroom('train.csv')
ggg_test <- vroom('test.csv')
ggg_missing <- vroom('trainWithMissingValues.csv')

colMeans(is.na(ggg_missing))
DataExplorer::plot_missing(ggg_missing)

ggg_recipe <- recipe(type ~ ., data=ggg_missing) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(has_soul), neighbors = 5) %>%
  step_impute_bag(rotting_flesh, impute_with = imp_vars(bone_length, hair_length, has_soul),
                  trees = 25) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(has_soul), neighbors = 5)

prep <- prep(ggg_recipe)
ggg_missing_baked <- bake(prep, new_data = ggg_missing)


rmse_vec(ggg_train[is.na(ggg_missing)], ggg_missing_baked[is.na(ggg_missing)])


quant_data <- ggg_missing %>%
  select(bone_length, rotting_flesh, hair_length, has_soul)

correlation_matrix <- cor(quant_data, use = "pairwise.complete.obs")
correlation_matrix
  

  


