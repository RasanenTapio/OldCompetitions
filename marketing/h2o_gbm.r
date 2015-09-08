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
predictors <- 2:1933
response <- 1934
puut <- 15
use_datasets <- "weight1"
results_file = "gbm_t71_d6_rate020"

# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")
	
# Load data
train_hex = h2o.importFile(localH2O, path = pathToTrain, header=TRUE)
train_valid = h2o.importFile(localH2O, path = pathToValid, header=TRUE)
test_hex = h2o.importFile(localH2O, path = pathToTest, key = "test_hex", header=TRUE)

# Convert to factor (note: h2o 3.0 requires this)
train_hex$target <- as.factor(train_hex$target)
train_valid$target <- as.factor(train_valid$target)
	
model_gbm1 <- h2o.gbm(y = response, x = predictors,
	training_frame = train_hex, validation = train_valid,
	distribution = "bernoulli",
	n.tree = 15, shrinkage = 0.2, interaction.depth = 6)
	
err <- auc(model_gbm1); err
		
# interaction.depth = c(6,8,7))
# depth 6, trees 45 : AUC 0.76848
# depth 6, trees 71 : AUC 0.77028 (converges at 56 trees?) (12+min)

# Bad interface on grid search! Use loop/script instead!
p_grid <- h2o.grid("deeplearning",
	y = response, x = predictors,
	training_frame = train_hex, validation_frame = train_valid,
	  activation= "RectifierWithDropout",
	  hidden=c(512,256,256),
	  balance_classes = TRUE,
	  classification_stop = -1,
	  epochs = 2,
	  hyper_params = list(l1 = c(0,1e-3,1e-5),
	  l2 = c(0,1e-3,1e-5)))

# Single model
model_deep1 <- h2o.deeplearning(y = response, x = predictors,
	training_frame = train_hex, validation_frame = train_valid,
	  activation= "RectifierWithDropout",
	  hidden=c(512,256,256),
	  balance_classes = TRUE,
	  classification_stop = -1,
	  epochs = 20,
	  l1 = 1e-5,
	  l2 = 1e-3)
	 
err2 <- auc(model_deep1); err2
	
# c(512,256,256): AUC: 0.70497 (2 epocsh)
# training samples = x?
# RectifierWithDropout: 0.72561 (0.5 per round) 4min - fast to train

# Grid Results:
# c(512,256,256): L1 0.00001 L2: 0.001: AUC: 0.724117
# c(512,256,256): L1 0 L2: 0: AUC: 0.722007
	# / epochs 5: 0.7243238
	# / epochs 20: 
		  
model_rf1 <- h2o.randomForest(y = response, x = predictors,
	 training_frame = train_hex, validation = train_valid,
	 classification = TRUE,
	 depth = 10,
	 ntree = 200,
	 seed = 69,
	 type = "BigData") # not yet h2o 3.0
	
# RF trees: 10+: AUC 0.74 (try to tune on 10-15 trees)

	
# Fit models and save results
pred_1 <- as.data.frame(h2o.predict(model_gbm1,test_hex))
idxs <- as.data.frame(test_hex$ID)

results <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1$X1))

density(results[,"target"])

write.table(results, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
quote = FALSE, col.names = TRUE, sep=",")

h2o.shutdown(localH2O)
print("done")