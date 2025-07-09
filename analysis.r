## ----globals -------------------------------------------------------------
library(mlr3verse)
library(mlr3pipelines)
library(glmnet) ## lasso and elastic net

## ----load.data -------------------------------------------------------------
data("ilpd", package = "mlr3data")

# summarise the dataset
str(ilpd)

# get a frequency table of the target variable
table(ilpd$diseased)

## ----prep.data -------------------------------------------------------------
## make female a logical vector
ilpd$female <- ilpd$gender=="Female"
ilpd$gender <- NULL

## Convert 'diseased' to a logical vector
ilpd$diseased <- {
            levels_map <- setNames(c(FALSE, TRUE), c("no", "yes"))
            as.logical(levels_map[as.character(ilpd$diseased)])
}

# summarise the final dataset
str(ilpd)
table(ilpd$diseased)

## ----task -------------------------------------------------------------
task = as_task_classif(
    x=ilpd,
    target="diseased",    
    id="liver disease")
task$col_roles$stratum = task$target_names

## ----split -------------------------------------------------------------
# Split manually:
set.seed(72340)
train_idx = sample(task$nrow, 0.8 * task$nrow)
test_idx = setdiff(seq_len(task$nrow), train_idx)

# Create train and test tasks:
task_train = task$clone()$filter(train_idx)
task_test  = task$clone()$filter(test_idx)


## ----rf -------------------------------------------------------------
# Random Forest pipeline
n = task_train$nrow
p = task_train$ncol

rf = (
    po("scale") %>>%
    lrn(
        "classif.ranger",
        oob.error=TRUE,
        importance="impurity",
        num.trees=100,
        mtry=floor(p/10),
        max.depth=floor(log2(16)), 
        min.node.size=16,
        predict_type="prob"))

## ----lasso -------------------------------------------------------------
# Lasso pipeline
lasso = (
    po("scale") %>>% 
    lrn(
        "classif.cv_glmnet",
        alpha=1,
        predict_type="prob"))

## ----elasticnet -------------------------------------------------------------
# Elastic Net pipeline
elasticnet = (
    po("scale") %>>% 
	auto_tuner(
			lrn(
				"classif.cv_glmnet",
				alpha=to_tune(0.1,0.9),
				predict_type="prob"),        
			tuner = tnr("grid_search"),
			resampling = rsmp("cv", folds = 5),
			measure = msr("classif.logloss"),
			term_evals = 20))
## ----xgboost -------------------------------------------------------------
# XGBoost pipeline
xgboost = (
    po("scale") %>>% 
	auto_tuner(
               lrn(
                   "classif.xgboost",
                   scale_pos_weight=1,  ## number samples/number positive samples                   
                   nrounds=100,#to_tune(100,200),## number boosting rounds
                   max_depth=4,#to_tune(3,4), ## max depth of trees
                   eta=to_tune(0.01,0.1), ## learning rate
                   subsample=0.9,#to_tune(0.8,0.9), ## proportion of samples supplied to a tree
                   colsample_bytree=0.9,#to_tune(0.8,0.9), ## proportion of features supplied to a tree
                   predict_type="prob"),
               tuner = tnr("grid_search"),
               resampling = rsmp("cv", folds = 5),
               measure = msr("classif.logloss"),
               term_evals = 20)
)

## ----bm -------------------------------------------------------------
library = c(rf, lasso, elasticnet, xgboost)
design = benchmark_grid(task_train, library, rsmp("cv",folds=3))
bm = benchmark(design)

## ----metrics -------------------------------------------------------------

# specify and calculate the measures to be used for evaluation
metrics = c("auc","acc","sensitivity","specificity",
    "precision","recall","fbeta")
metrics = paste("classif", metrics, sep=".")
metrics = lapply(metrics, msr)

# print the measures
bm$aggregate(measure=metrics)

## ----plots -------------------------------------------------------------

autoplot(bm, measure=msr("classif.auc"))

autoplot(bm, type = "roc")

autoplot(bm, type = "prc")