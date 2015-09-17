# Features from Python: Butter filter and haar wavelet

setwd("C:/marketingdata")

#### h2o cluster ####
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="8g")
set.seed(6)

# Function to shorten h2o.auc call
auc <- function(model_in){
	err <- h2o.auc( h2o.performance(model_in, train_valid) )
	return(err)
}

# Model parameters
predictors <- c(2:1933) # predictors <- c(2:1933, 1935:1938)
response <- 1934
use_datasets <- "weight2a"
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

# datasets A 1:10, i = validation set, A / i = training set
# Split to 10 parts
rat <- rep(0.333,2)
train_hex_split <- h2o.splitFrame(train_hex, ratios = rat)


# verify
head(train_hex_split[[3]]$target)

for (i in 1:3) {

	# fit model to sample
	model_gbm1 <- h2o.gbm(y = response, x = predictors,
		model_id = paste("gbm",i,sep=""),
		training_frame = train_hex_split[[i]], validation = train_valid, # train and valid
		distribution = "bernoulli",
		#nfolds = 3,
		ntrees = 78, learn_rate = 0.2, max_depth = 6) # best: 78

	err <- auc(h2o.getModel(paste("gbm",i,sep=""))); print(err)
}

### Predict for test set
# Fit models and save results
test_hex = h2o.importFile(localH2O, path = pathToTest, header=TRUE)

# Predict
idxs <- as.data.frame(test_hex$ID)

results <- data.frame(idx = 1:dim(test_hex)[1])
results <- results[,-1]
models <- 3
for (i in 1:models){
	results <- cbind(results, as.data.frame(h2o.predict(h2o.getModel(paste("gbm",i,sep="")),test_hex)$p1))
}
names(results) <- paste("model",1:models)

res_avg <- rowMeans(results)

results_file <- "gbm_m3_split3_val_t78"

submission <- as.matrix(data.frame(ID = as.character(idxs$ID), target = res_avg))

write.table(submission, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
quote = FALSE, col.names = TRUE, sep=",")

# Stacking (save 10th frame for stacking)

# models to get:

# construct loop for this names(d)[names(d)=="beta"] <- "two"
train_valid$temp <- h2o.predict(h2o.getModel("gbm1"),train_valid)$predict # get p1 and predict
# rename model_p1 model_c1
train_valid$p1 <- h2o.predict(h2o.getModel("gbm1"),train_valid)$p1


model_stack = h2o.naiveBayes(x = c(paste("p",1:10,sep="")), y = response, model_id = "stack1",
		 training_frame = train_valid)

model_stack <- h2o.deeplearning(y = response, x = c(paste("c",1:10,sep="")), model_id = "stack1",
	training_frame = train_valid,
	  activation= "Tanh",
	  hidden = c(66,33,33,22),
	  #loss = "CrossEntropy",
	  #train_samples_per_iteration = 5000,
	  epochs = 50,
	  #hidden_dropout_ratios = c(0.5,0.5,0.5),
	  seed = 12)
		 
auc(h2o.getModel("stack1"))
# Bayes: 0.7506003 on training data
# NN:

h2o.shutdown(localH2O)