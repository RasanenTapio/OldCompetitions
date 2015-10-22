# Working directory
setwd("C:/marketingdata")

#### h2o cluster ####
library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="8g")
set.seed(19)
use_datasets <- "featuredate6"

# Function to shorten h2o.auc call
auc_stack <- function(model_in, data_in){
	err <- h2o.auc( h2o.performance(model_in, data_in) )
	return(err)
}


# Load data and training data and set labels
pathToTrain = paste("C:/marketingdata/",use_datasets,"/train.csv", sep = "")
pathToValid = paste("C:/marketingdata/",use_datasets,"/valid.csv", sep = "")
pathToTest = paste("C:/marketingdata/",use_datasets,"/test.csv", sep = "")

# Load data
#train_hex = h2o.importFile(localH2O, path = pathToTrain, header=TRUE)
train_valid = h2o.importFile(localH2O, path = pathToValid, header=TRUE)

# Convert to factor (note: h2o 3.0 requires this)
#train_hex$target <- as.factor(train_hex$target)
train_valid$target <- as.factor(train_valid$target)

# models to get:
models <- read.csv("featuredate6/stacking/models.csv", stringsAsFactors = FALSE)$modelmeta
spath <- 'featuredate6/stacking/'

#stacking_class <- matrix(0,dim(train_valid),length(models)+1)
stacking_prob <- matrix(0,dim(train_valid),length(models)+1)

for (i in 1:length(models)){
	m <- models[i]
	#stacking_class[,i] <- read.csv(paste(spath,models[i], "_vc.csv", sep =""), stringsAsFactors = FALSE)$target
	stacking_prob[,i] <- read.csv(paste(spath,models[i], "_vp.csv", sep =""), stringsAsFactors = FALSE)$target
}

# Upload to cluster
#stacking_class[,length(models)+1] <- as.vector(as.data.frame(train_valid$target)$target)
stacking_prob[,length(models)+1] <- as.vector(as.data.frame(train_valid$target)$target)
#colnames(stacking_class) <- c(paste("Var",1:length(models),sep=""),"target")
colnames(stacking_prob) <- c(paste("Var",1:length(models),sep=""),"target")

stacking1_hex <- as.h2o(stacking_prob)
#stacking2_hex <- as.h2o(stacking_class)
stacking1_hex$target <- as.factor(stacking1_hex$target)
#stacking2_hex$target <- as.factor(stacking2_hex$target)

# Split
stack_hex_split <- h2o.splitFrame(stacking1_hex, ratios = 0.7)

response_s <- which(names(stacking1_hex) == "target")
predictors_s <- 1:5 # available 

# 
model_stack = h2o.naiveBayes(x = predictors_s, y = response_s, model_id = "stack1",
		 training_frame = stack_hex_split[[1]])

auc_stack(model_stack, stack_hex_split[[2]]) 
# 0.7555 on
# 0.7555903
# 0.7546265 on adding 2nd rf model
# 0.7536499 on adding 3rd rf model
# 8:18 : 0.7733665
# 10:18 : 0.7798353
# 12:18 : 0.7852689
# 15:18 : 0.7907206
# 16:18 : 0.7914214 # conclusion: train more good models
# 17:18 : 0.7852689
# 10:20 # 0.939758 DA FUK?

# f6: 4m:  0.7902411

model_stack1 <- h2o.deeplearning(y = response_s, x = predictors_s, model_id = "stack1",
	training_frame = stack_hex_split[[1]], validation = stack_hex_split[[2]],
	  activation= "Tanh",
	  hidden = c(66,33,33), # best on hidden = c(66,33,33,22),
	  #loss = "CrossEntropy",
	  epochs = 50,
	  hidden_dropout_ratios = c(0,0,0),
	  seed = 19)


auc2 <- auc_stack(model_stack1, stack_hex_split[[2]]); auc2

# f6: 4m: c(66,33,33): 0.7923382

# 0.7476609 on p
# 0.7525552 on p
# 0.7409493 on adding 2nd rf model
# 0.7460762 on adding 3rd rf model # LB AUC: 0.72
# 0.7893497 on 18 models
# 9:18: 0.7887448 on  models
# 12:18: 0.7916486 (0.7/0.3 split: 0.7931127 on valid)
# 10:19: 0.7920637 (no LB improvement)
# 11:19: 0.7916331
# 11:20: 0.7892738
# 10:20:  0.7912938

# c(77,33,33) network:
# 10:19 - 0.7930042

# different settings: 

# Bayes: 0.7506003 on training data
# NN:

#### Save stacked

# Test dimension 145232   1979

# Get test
#test_class <- matrix(0,145232,length(models))
test_prob <- matrix(0,145232,length(models))
spath <- 'results/'

# Data starts from model 10
for (i in 1:length(models)){
	m <- models[i]
	#test_class[,i] <- read.csv(paste(spath,models[i], "_tc.csv", sep =""), stringsAsFactors = FALSE)$target
	test_prob[,i] <- read.csv(paste(spath,models[i], ".csv", sep =""), stringsAsFactors = FALSE)$target
}
#colnames(test_class) <- paste("Var",1:length(models),sep="")
colnames(test_prob) <- paste("Var",1:length(models),sep="")
testprob_hex <- as.h2o(test_prob) # this explains it..
#testclass_hex <- as.h2o(stacking_class)

test_hex = h2o.importFile(localH2O, path = pathToTest, header=TRUE)

results_file <- "stack_f6_m4_deep1"

pred_1 <- as.data.frame(h2o.predict(model_stack,testprob_hex))$p1
#pred_2 <- as.data.frame(h2o.predict(model_stack2,testprob_hex))$p1
#preds <- (pred_1*auc1 + pred_2*auc2)/(auc1+auc2)
idxs <- as.data.frame(test_hex$ID)

results <- as.matrix(data.frame(ID = as.character(idxs$ID), target = pred_1))

write.table(results, file = paste('results/', results_file,'.csv', sep = ""), row.names = FALSE,
quote = FALSE, col.names = TRUE, sep=",")

h2o.shutdown(localH2O)