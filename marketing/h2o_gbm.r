setwd("C:/marketingdata")

#### h2o cluster ####
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="8g")
use_datasets <- "featuredate3"

# Function to shorten h2o.auc call
auc <- function(model_in,data_in){
	err <- h2o.auc( h2o.performance(model_in, data_in) )
	return(err)
}

# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")

# Variable importance
#importance <- read.csv(paste("C:/marketingdata/",use_datasets,"/importance.csv", sep = ""))
#importance <- as.vector(as.matrix(importance))

# Load data
train_hex = h2o.importFile(localH2O, path = pathToTrain, header=TRUE)
train_valid = h2o.importFile(localH2O, path = pathToValid, header=TRUE)

# Convert to factor (note: h2o 3.0 requires this)
train_hex$target <- as.factor(train_hex$target)
train_valid$target <- as.factor(train_valid$target)

##### MODELLING
# Model parameters
response <- which(names(train_hex)=="target") # 1919
#predictors <- predictors <- names(train_hex)[!(names(train_hex) %in% c("target", "ID"))]
predictors <- names(train_hex)[!(names(train_hex) %in%
	c("target", "ID","VAR_0073_date","VAR_0073_month","VAR_0073_wd","VAR_0073_year"))] # without VAR_0073_date

## Additional:
rat <- rep(0.3,3)
train_hex_split <- h2o.splitFrame(train_hex, ratios = 0.85)

#use_param <- c(7,9,10)
use_param <- 1
for(i in 1:length(use_param)){
	print(paste("parameter/split:",use_param[i]))

	model_gbm1 <- h2o.gbm(y = response, x = predictors, model_id = paste("gbm",i,sep=""),
		training_frame = train_hex_split[[i]], validation = train_hex_split[[2]], #validation = train_valid,
		distribution = "bernoulli",
		ntrees = 151, learn_rate = 0.03, max_depth = 10, seed = 11) # set seed
		
	err <- auc(model_gbm1,train_hex_split[[2]]); print(err)
}
#				0.15 valid	: 0.10 validation
# d : old feat  : new feat  : without VAR_0073	: with VAR_0073
# 6 : --------- : --------  : 0.7713394 
# 7 : --------- : --------  : 0.7727186 
# 8 : 0.773803  : 0.7741644 : 0.7737159 : 0.7732035 (old=0.7740866?)
# 9 : --------- : --------- : 0.7741497 : 0.7744719
# 10 : -------- : --------- : 0.7760306 best :  0.7739738 <- overfits training data: VAR_0073_data has too high relative importance!
# 11 : -------- : --------- : 0.7733163
# 12 : -------- : --------- : 0.7751893

# 10: 50 trees: 26min -> 115 trees: 0.7789472
# 10: 115 trees, rate 0.03: 0.7756567 (can still improve)

# when does rate 0.05 converge?
# 0.7741312 on 112 trees, depth 6
# 0.7778351 on 112 trees, depth 8 / 46min --- best?

# 112 trees, rate 0.01 or 0.025 / 49min -- can improve (0.7589593 and 0.769661)

# new features, rate 0.2
# depth 6, trees 78: AUC 0.773107
# depth 6, trees 96: AUC 0.7744199 (1980 dimension) / 23 min -- broken!

# Single model
model_deep1 <- h2o.deeplearning(y = response, x = predictors, model_id = "deep1",
	training_frame = train_hex_split[[1]], validation_frame = train_hex_split[[4]],
	  activation= "RectifierWithDropout",
	  hidden = c(512,256,256),
	  #balance_classes = TRUE,
	  epochs = 25)
	  #l1 = 1e-5,
	  #l2 = 1e-3
	 
err2 <- auc(model_deep1); err2

# New date data: 0.7240608 on 2.5 epochs

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
	 training_frame = train_hex_split[[3]], validation_frame = train_hex_split[[4]],
	 max_depth = 12,
	 ntree = 50, # 50
	 seed = 69,
	 type = "BigData",
	 balance_classes = TRUE)

# balance_classes, nfold
	 
err3 <- auc(model_rf1,train_hex_split[[4]]); err3

# New features: 0.749001

# Tune on 20 (50?) trees (weight1):
# max depth 12 : AUC 0.7386302 ### best: 0.7451361 on 81 trees ### 0.746901 on 151 trees
# new features
# depth 11, trees 50 : AUC 0.7500931
# depth 11, trees 151 : AUC 0.7509202
# depth 12, trees 50 : AUC 0.7513113 (old features: 0.7519797)

# Too many features? use features generated with importance = TRUE?
bayes_predictors <- c("VAR_0137","VAR_0795","VAR_0088","VAR_1329",
	"VAR_1127","VAR_0080","VAR_0540","VAR_1377","VAR_1747","VAR_0758","VAR_0241",
	"VAR_0886","VAR_1791","VAR_0970","VAR_0505","VAR_0503","VAR_1126","VAR_1380",
	"VAR_1100","VAR_1042","VAR_1029","VAR_0719","VAR_0969","VAR_0005","VAR_0881",
	"VAR_1823","VAR_0896","VAR_0004","VAR_0707","VAR_1020","VAR_0656","VAR_0716",
	"VAR_1399","VAR_0227","VAR_0613","VAR_0579","VAR_1398","VAR_1934")
#bayes_predictors[1:20]
# remove and add predictors until AUC no longer improves
model_nb = h2o.naiveBayes(x = bayes_predictors[1:15], y = response,
		 training_frame = train_hex)

errb <- auc(model_nb); errb # 0.6855 # 1:20 best features: AUC  0.7008633
		 
# Fit models and save results
test_hex = h2o.importFile(localH2O, path = pathToTest, header=TRUE)

# Predict
results_file = "gbm_t151_d10_r03_f1"
model_to_get <- h2o.getModel("gbm3")
write_results <- TRUE

# Function for creating metadata for stacking
# As function: parameters: use_datasets, results_file, model_to_get, write_results
	pred_1 <- as.data.frame(h2o.predict(model_to_get,test_hex))
	idxs <- as.data.frame(test_hex$ID)

	results <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1$p1))

	### For single submission
	if (write_results == TRUE){
		write.table(results, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
			quote = FALSE, col.names = TRUE, sep=",")
	}

	### For stacking ensemble results_file_valid_stack
	idxs_valid <- as.data.frame(train_valid$ID)
	pred_v <- as.data.frame(h2o.predict(model_to_get,train_valid))

	results_class <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1$predict))
	valid_p1 <- as.matrix(data.frame(ID = as.character(idxs_valid$ID), target = pred_v$p1))
	valid_class <- as.matrix(data.frame(ID = as.character(idxs_valid$ID), target = pred_v$predict))

	# Save p1 for test
	write.table(results, file = paste(use_datasets,'/stacking/', results_file,'_tp','.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")

	# Save class for test
	write.table(results_class, file = paste(use_datasets,'/stacking/', results_file,'_tc','.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")

	# Save p1 for valid
	write.table(valid_p1, file = paste(use_datasets,'/stacking/', results_file,'_vp','.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")

	# Save class for valid
	write.table(valid_class, file = paste(use_datasets,'/stacking/', results_file,'_vc','.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")
	
	### Append saved name to file
	write.table(data.frame(modelmeta = results_file), file = paste(use_datasets,'/stacking/models.csv', sep = ""), row.names = FALSE,
		quote = FALSE, col.names = FALSE, sep=",", append = TRUE)

## Done
h2o.shutdown(localH2O)
print("done")