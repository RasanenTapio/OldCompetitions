# Features from Python: Butter filter and haar wavelet

setwd("C:/marketingdata")

#### h2o cluster ####
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="8g")

# Function to shorten h2o.auc call
auc <- function(model_in){
	err <- h2o.auc( h2o.performance(model_in, train_valid) )
	return(err)
}

# Model parameters
predictors <- c(2:1933)
#predictors <- c(2:1933, 1935:1980) # exclude X1_year #
response <- 1934
use_datasets <- "weight1a"
results_file = "gbm_t67_d6_h2o_datetest1"

# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")

# Variable importance
importance <- read.csv(paste("C:/marketingdata/",use_datasets,"/importance.csv", sep = ""))
importance <- as.vector(as.matrix(importance))

# Load data
train_hex = h2o.importFile(localH2O, path = pathToTrain, header=TRUE)
train_valid = h2o.importFile(localH2O, path = pathToValid, header=TRUE)

# Convert to factor (note: h2o 3.0 requires this)
train_hex$target <- as.factor(train_hex$target)
train_valid$target <- as.factor(train_valid$target)

use_param <- c(6,7,8,9,10)
for(i in 1:length(use_param){
	print(paste("parameter:",use_param[i]))

	model_gbm1 <- h2o.gbm(y = response, x = predictors, model_id = paste("gbm",i,sep=""),
		training_frame = train_hex, validation = train_valid,
		distribution = "bernoulli",
		ntrees = 50, learn_rate = 0.10, max_depth = 8)
		
	err <- auc(model_gbm1); print(err)
}

# interaction.depth = c(6,8,7))

# when does rate 0.05 converge?
# 0.7741312 on 112 trees, depth 6
# 0.7778351 on 112 trees, depth 8 / 46min --- best?

# 112 trees, rate 0.01 or 0.25 / 49min -- can improve (0.7589593 and 0.769661)

# new features, rate 0.2
# depth 6, trees 78: AUC 0.773107
# depth 6, trees 96: AUC 0.7744199 (1980 dimension) / 23 min -- broken!

# Single model
model_deep1 <- h2o.deeplearning(y = response, x = importance, model_id = "deep1",
	training_frame = train_hex, validation_frame = train_valid,
	  activation= "RectifierWithDropout",
	  hidden = c(512,256,256),
	  #balance_classes = TRUE,
	  classification_stop = -1,
	  epochs = 2.5,
	  l1 = 1e-5,
	  l2 = 1e-3) #
	 
err2 <- auc(model_deep1); err2
	
# 2 epochs
# c(512,256,256): AUC: 0.70497
# c(512,256,256), important features: AUC: 0.7198006
# hidden = c(666,441,296) : AUC 0.7049955


# training samples = x?
# RectifierWithDropout: 0.72561 (0.5 per round) 4min - fast to train

# Grid Results / epochs 2:
# c(512,256,256): L1 0.00001 L2: 0.001: AUC: 0.724117
# c(512,256,256): L1 0 L2: 0: AUC: 0.722007
	# / epochs 5: 0.7243238
	# / epochs 20: bad
# c(1024,512,256): AUC: 0.7225449
# c(1281,986,521): AUC: 0.7252116 (epochs 5)
# c(1281,986,521): AUC: 0.7312193 (epochs 25)

# New data, 10 epochs
# c(512,256,256) : AUC 0.7121637 / 2 epochs

model_rf1 <- h2o.randomForest(y = response, x = predictors, model_id = "rf1",
	 training_frame = train_hex, validation_frame = train_valid,
	 max_depth = 12,
	 ntree = 50, # 50
	 seed = 69,
	 type = "BigData",
	 balance_classes = TRUE)

# balance_classes, nfold
	 
err3 <- auc(model_rf1); err3

# Tune on 20 (50?) trees (weight1):
# max depth 12 : AUC 0.7386302 ### best: 0.7451361 on 81 trees ### 0.746901 on 151 trees
# new features
# depth 11, trees 50 : AUC 0.7500931
# depth 11, trees 151 : AUC 0.7509202
# depth 12, trees 50 : AUC 0.7513113 (old features: 0.7519797)

# Too many features? use features generated with importance = TRUE?
bayes_predictors <- c("VAR_0137","VAR_0795","VAR_0088","VAR_0073","VAR_1329",
	"VAR_1127","VAR_0080","VAR_0540","VAR_1377","VAR_1747","VAR_0758","VAR_0241",
	"VAR_0886","VAR_1791","VAR_0970","VAR_0505","VAR_0503","VAR_1126","VAR_1380",
	"VAR_1100","VAR_1042","VAR_1029","VAR_0719","VAR_0969","VAR_0005","VAR_0881",
	"VAR_1823","VAR_0896","VAR_0004","VAR_0707","VAR_1020","VAR_0656","VAR_0716",
	"VAR_1399","VAR_0227","VAR_0613","VAR_0579","VAR_1398","VAR_1934")

# remove and add predictors until AUC no longer improves
model_nb = h2o.naiveBayes(x = bayes_predictors[1:20], y = response,
		 training_frame = train_hex)

errb <- auc(model_nb); errb # 0.6855 # 1:20 best features: AUC  0.7008633
		 
# Fit models and save results
test_hex = h2o.importFile(localH2O, path = pathToTest, header=TRUE)

# Predict
pred_1 <- as.data.frame(h2o.predict(model_gbm1,test_hex))
idxs <- as.data.frame(test_hex$ID)

results <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1$p1))

density(as.numeric(results[,"target"]))

write.table(results, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
quote = FALSE, col.names = TRUE, sep=",")

h2o.shutdown(localH2O)
print("done")