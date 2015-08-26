# Features from Python: Butter filter and haar wavelet

setwd("C:/eegdata")

#### h2o cluster ####

library(h2o)
localH2O <- h2o.init(nthread=4,Xmx="8g")

predictors <- 8:183 # predictors <- 8:dim(train_hex)[2]
puut <- 10
use_datasets <- "datad" #datad datas
results_file = "gbm_t30_feat183_haar"

# Loop over responses 2-7 for every subject in 1:12
for (subject in 1:12) {
	# Load data and training data and set labels
	pathToTrain = paste("C:/eegdata/",use_datasets,"/train",subject,".csv", sep = "")
	pathToTest = paste("C:/eegdata/",use_datasets,"/test",subject,".csv", sep = "")
	
	# Load data
	train_hex = h2o.importFile(localH2O, path = pathToTrain, key = "train_hex", header=TRUE)
	test_hex = h2o.importFile(localH2O, path = pathToTest, key = "test_hex", header=TRUE)

	# Set labels
	names_train <- c("id","HandStart","FirstDigitTouch","BothStartLoadPhase","LiftOff","Replace","BothReleased",
		paste("X", 1:(dim(train_hex)[2]-7), sep = ""))
	names(train_hex) <- names_train
	print(paste("Subject",subject,"train data loaded", sep = " "))
	names(test_hex) <- c("id",paste("X", 1:(dim(test_hex)[2]-1), sep = ""))
	print(paste("Subject",subject,"test data loaded", sep = " "))

	# fit 6 models
	
	print(paste("Subject",subject,"HandStart", sep = " "))
	model_gbm1 <- h2o.gbm(y = 2, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut + 5, shrinkage = 0.05, interaction.depth = 13)
	
		#importance = TRUE try!
	print(paste("Subject",subject,"FirstDigitTouch", sep = " "))
	model_gbm2 <- h2o.gbm(y = 3, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut, shrinkage = c(0.05, 0.1, 0.2), interaction.depth = 13)
		
	# Grid search, run model_gbm2
	stop

	print(paste("Subject",subject,"BothStartLoadPhase", sep = " "))
	model_gbm3 <- h2o.gbm(y = 4, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut, shrinkage = 0.05, interaction.depth = 13)

	print(paste("Subject",subject,"LiftOff", sep = " "))
	model_gbm4 <- h2o.gbm(y = 5, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut, shrinkage = 0.05, interaction.depth = 13)
	
	print(paste("Subject",subject,"Replace", sep = " "))
	model_gbm5 <- h2o.gbm(y = 6, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut, shrinkage = 0.05, interaction.depth = 13)
	
	print(paste("Subject",subject,"BothReleased", sep = " "))
	model_gbm6 <- h2o.gbm(y = 7, x = predictors,
		data = train_hex,
		distribution = "bernoulli",
		n.tree = puut, shrinkage = 0.05, interaction.depth = 13)

	# Fit models
	pred_1 <- as.data.frame(h2o.predict(model_gbm1,test_hex))
	pred_2 <- as.data.frame(h2o.predict(model_gbm2,test_hex))
	pred_3 <- as.data.frame(h2o.predict(model_gbm3,test_hex))
	pred_4 <- as.data.frame(h2o.predict(model_gbm4,test_hex))
	pred_5 <- as.data.frame(h2o.predict(model_gbm5,test_hex))
	pred_6 <- as.data.frame(h2o.predict(model_gbm6,test_hex))
	print(paste("Subject",subject,"models fitted", sep = " "))
	# Append to results file
	idxs <- as.data.frame(test_hex$id)

	results = as.matrix(data.frame(
		id = as.data.frame(test_hex$id),
		HandStart = round(pred_1$X1,4),
		FirstDigitTouch = round(pred_2$X1,4),
		BothStartLoadPhase = round(pred_3$X1,4),
		LiftOff = round(pred_4$X1,4),
		Replace = round(pred_5$X1,4),
		BothReleased = round(pred_6$X1,4)))
		
	print(paste("Subject",subject,"results formed", sep = " "))
	
	if (subject == 1) {
	write.table(results, file = paste('results_gbm/',results_file,".csv", sep = ""),
		row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",", append = TRUE)
		print(paste("Subject",subject,"results saved with header", sep = " "))
	}else{
	write.table(results, file = paste('results_gbm/',results_file,".csv", sep = ""),
		row.names = FALSE, quote = FALSE, col.names = FALSE, sep=",", append = TRUE)
		print(paste("Subject",subject,"results appended", sep = " "))
	}
}

h2o.shutdown(localH2O)
print("done")