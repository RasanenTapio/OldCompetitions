setwd("C:/marketingdata")

#### h2o cluster ####
#library(data.table) #some sort of memory leak problem?
library(readr)
library(xgboost)
use_datasets <- "featuredate3"
use_split <- 1 # split to be used as validation frame

# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")

train <- read_csv(pathToTrain)
val <- read_csv(pathToValid)

# Combine
# if use splits:
#train <- rbind(train, val)
#val=val[1:3,]

### From Kaggle scripts -- verify this!
train.unique.count=lapply(train, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const=names(train.unique.count_1)
delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

#VARS to delete
#safe to remove VARS with 56, 89 and 918 NA's as they are covered by other VARS
print(length(c(delete_const,delete_NA56,delete_NA89,delete_NA918)))

# KORJAA TÄTÄ: poista vain vakiot? drop id
train=train[,!(names(train) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918))]
#	"VAR_0073_date","VAR_0073_month","VAR_0073_wd","VAR_0073_year"))]

feature.names <- names(train)[2:ncol(train)]
feature.names <- feature.names[-which(feature.names == "target")]
length(feature.names) # 1971 #1967 # 1923 features??

# Set seed before sample
#set.seed(7)
#sample_f <- sample(dim(train)[1], dim(train)[1]*0.9)
if (use_split) {
	sample_split <- split(1:dim(train)[1], 1:10)[[1]]

	dtrain <- xgb.DMatrix(data.matrix(train[-sample_split,feature.names]), label=train$target[-sample_split])

	dval <- xgb.DMatrix(data.matrix(train[sample_split,feature.names]), label=train$target[sample_split])
	train=train[1:3,]
}
# else: tuning
dtrain <- xgb.DMatrix(data.matrix(train[,feature.names]), label=train$target)
dval <- xgb.DMatrix(data.matrix(val[,feature.names]), label=val$target)

val=val[1:3,]
train=train[1:3,]
gc()

watchlist <- watchlist <- list(eval = dval, train = dtrain)

# best: eta 0.01, depth 12, subsample 0.7, colsample_bytree 0.8
param <- list(  objective           = "binary:logistic", 
                eta                 = 0.025,
                max_depth           = 12,
                subsample           = 0.7, # 0.7
                colsample_bytree    = 0.8, # 0.7
                eval_metric         = "auc",
                scale_pos_weight	= 1 # new feature
                )
## # scale weight of positive examples
##param['scale_pos_weight'] = sum_wneg/sum_wpos
# gamme = ?

# should be better than 0.78960 and AUC 0.784889 on training set.

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1100, #300, #280, #125, #250, # changed from 300
                    verbose             = 1,
					nthread				= 4,
                    #early.stop.round    = 10,
                    watchlist           = watchlist,
                    maximize            = TRUE)


#dtrain=0
#gc()

#dval=0
#gc()

test <- read_csv(pathToTest)
test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918,"VAR_0073"))]

submission <- data.frame(ID=test$ID)
submission$target <- NA 
for (rows in split(1:nrow(test), ceiling((1:nrow(test))/10000))) {
    submission[rows, "target"] <- predict(clf, data.matrix(test[rows,feature.names]))
 
}


cat("saving the submission file\n")
write_csv(submission, "C:/marketingdata/results/xgb_r01_d12_r1000.csv")

#### STACKING/METALEARNER PREDICTORS
### Only if validation data has not been used:
val <- read_csv(pathToValid)

submission_v <- data.frame(ID=val$ID)
submission_v$target <- NA 


submission_v[, "target"] <- predict(clf, data.matrix(val[,feature.names]))

cat("saving the submission file\n")
write_csv(submission_v, "C:/marketingdata/featuredate3/stacking/xgb_r01_d12_r1100_vp.csv")

write.table(data.frame(modelmeta = "xgb_r01_d10_r1000"), file = paste('C:/marketingdata/featuredate3/stacking/models_p.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = FALSE, sep=",", append = TRUE)



# Save prediction on validation set for stacking!