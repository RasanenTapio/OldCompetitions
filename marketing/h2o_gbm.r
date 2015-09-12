# Features from Python: Butter filter and haar wavelet

setwd("C:/marketingdata")

#### h2o cluster ####
library(h2o)
localH2O <- h2o.init(nthread=8,Xmx="8g")

# Function to shorten h2o.auc call
auc <- function(model_in){
	err <- h2o.auc( h2o.performance(model_in, train_valid) )
	return(err)
}

# Model parameters
predictors <- 2:1933
response <- 1934
puut <- 15
use_datasets <- "weight5"
results_file = "gbm_t81_d12_h2o"

# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")
	
# Load data
train_hex = h2o.importFile(localH2O, path = pathToTrain, header=TRUE)
train_valid = h2o.importFile(localH2O, path = pathToValid, header=TRUE)

# Convert to factor (note: h2o 3.0 requires this)
train_hex$target <- as.factor(train_hex$target)
train_valid$target <- as.factor(train_valid$target)
	
model_gbm1 <- h2o.gbm(y = response, x = predictors, model_id = "nalle",
	training_frame = train_hex, validation = train_valid,
	distribution = "bernoulli",
	ntrees = 71, learn_rate = 0.2, max_depth = 6)

err <- auc(model_gbm1); err
print("Get Model") # getModel "nalle"

# interaction.depth = c(6,8,7))
# depth 6, trees 45 : AUC 0.76848
# depth 6, trees 71 : AUC 0.77028 (converges at 56 trees?) (12+min)
# depth 6, trees 15 : AUC 0.7693006

# Single model
model_deep1 <- h2o.deeplearning(y = response, x = predictors, model_id = "nallen",
	training_frame = train_hex, validation_frame = train_valid,
	  activation= "RectifierWithDropout",
	  hidden=c(1281,986,521),
	  distribution = "bernoulli",
	  balance_classes = TRUE,
	  classification_stop = -1,
	  epochs = 5,
	  rate = 0.05, # tune this?
	  l1 = 1e-5,
	  l2 = 1e-3)
	 
err2 <- auc(model_deep1); err2

	
# c(512,256,256): AUC: 0.70497 (2 epocsh)
# training samples = x?
# RectifierWithDropout: 0.72561 (0.5 per round) 4min - fast to train

# Grid Results / epochs 2:
# c(512,256,256): L1 0.00001 L2: 0.001: AUC: 0.724117
# c(512,256,256): L1 0 L2: 0: AUC: 0.722007
	# / epochs 5: 0.7243238
	# / epochs 20: bad
# c(1024,512,256): AUC: 0.7225449
# c(1281,986,521): AUC: 0.7252116 (epochs 5)
# c(1281,986,521): AUC: xxx (epochs 25)

model_rf1 <- h2o.randomForest(y = response, x = predictors, model_id = "nallerf",
	 training_frame = train_hex, validation = train_valid,
	 classification = TRUE,
	 max_depth = 12,
	 ntree = 151,
	 seed = 69,
	 type = "BigData")

# balance_classes, nfold
	 
err3 <- auc(model_rf1); err3
	
# Tune on 20 trees:
# max depth 10 : AUC 0.7373466
# max depth 11 : AUC 0.7382244
# max depth 12 : AUC 0.7386302 ### best: 0.7451361 on 81 trees ### 0.746901 on 151 trees
# max depth 13 : AUC 0.7383014
# max depth 15 : AUC 0.737508
# max depth 20 : AUC  0.7324021
# max depth 25 : AUC 0.7271571
	
# Fit models and save results
test_hex = h2o.importFile(localH2O, path = pathToTest, header=TRUE)

# Predict
pred_1 <- as.data.frame(h2o.predict(model_rf1,test_hex))
idxs <- as.data.frame(test_hex$ID)

results <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1$p1))

density(as.numeric(results[,"target"]))

write.table(results, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
quote = FALSE, col.names = TRUE, sep=",")

h2o.shutdown(localH2O)
print("done")