setwd("C:/mutualdata")
library(h2o)
library(plyr)

set.seed(10)

# Load data
train <- read.csv("mutualdata/train.csv", header = TRUE)
test <- read.csv("mutualdata/test.csv", header = TRUE)

# Drop id
train <- train[, -1]

#for(i in 2:dim(train)[2]) {
#	train[,i] <- as.factor(train[,i])
#}

#for(i in 2:dim(test)[2]) {
#	test[,i] <- as.factor(test[,i])
#}

# Or this:
train$T1_V4<-as.factor(train$T1_V4)
train$T1_V5<-as.factor(train$T1_V5)
train$T1_V6<-as.factor(train$T1_V6)
train$T1_V7<-as.factor(train$T1_V7)
train$T1_V8<-as.factor(train$T1_V8)
train$T1_V9<-as.factor(train$T1_V9)
train$T1_V11<-as.factor(train$T1_V11)
train$T1_V12<-as.factor(train$T1_V12)
train$T1_V15<-as.factor(train$T1_V15)
train$T1_V16<-as.factor(train$T1_V16)
train$T1_V17<-as.factor(train$T1_V17)
train$T2_V3<-as.factor(train$T2_V3)
train$T2_V5<-as.factor(train$T2_V5)
train$T2_V11<-as.factor(train$T2_V11)
train$T2_V12<-as.factor(train$T2_V12)
train$T2_V13<-as.factor(train$T2_V13)

test$T1_V4<-as.factor(test$T1_V4)
test$T1_V5<-as.factor(test$T1_V5)
test$T1_V6<-as.factor(test$T1_V6)
test$T1_V7<-as.factor(test$T1_V7)
test$T1_V8<-as.factor(test$T1_V8)
test$T1_V9<-as.factor(test$T1_V9)
test$T1_V11<-as.factor(test$T1_V11)
test$T1_V12<-as.factor(test$T1_V12)
test$T1_V15<-as.factor(test$T1_V15)
test$T1_V16<-as.factor(test$T1_V16)
test$T1_V17<-as.factor(test$T1_V17)
test$T2_V3<-as.factor(test$T2_V3)
test$T2_V5<-as.factor(test$T2_V5)
test$T2_V11<-as.factor(test$T2_V11)
test$T2_V12<-as.factor(test$T2_V12)
test$T2_V13<-as.factor(test$T2_V13)

train<-as.data.frame(train)
test<-as.data.frame(test)

# if sample:
#otos <- sample(1:dim(train)[1], dim(train)[1]*0.1)

# Hazard or log+1 hazard?
train$Hazard <- log(train$Hazard + 1)

train_a <- split(train, sample(rep(1:2, dim(train)[1]))) # try sampling with replacement
train1 <- train_a[[1]]
train2 <- train_a[[2]]


# kokeile 1/16 root
#train$Hazard <- (train$Hazard)^(1/16)

#trainsmall <- train[-otos,]
#valid <- train[otos,]
# cluster
localH2O <- h2o.init(nthread=4,Xmx="10g")

h2o_test <- as.h2o(localH2O,test)

# Parameters
response <- 1
predictors <- 2:33
looppeja <- 10

# Create results frame
results <- read.csv("sample_submission.csv", header = TRUE)
append_results1 <- results
append_results2 <- results

# Some models (Gradient boosted trees)
for(i in 1:looppeja){

	# split data
	train_a <- split(train, sample(rep(1:2, dim(train)[1]))) # try sampling with replacement
	train1 <- train_a[[1]]
	train2 <- train_a[[2]]
	# or 2 samples with or withot replacement
	
	h2o_train1 <- as.h2o(localH2O,train1)
	h2o_train2 <- as.h2o(localH2O,train2)

	model6 <- h2o.gbm(y = response, x = predictors,
		data = h2o_train1,
		distribution = "gaussian",
		n.tree = 150, shrinkage = 0.2, interaction.depth = 15)
		
	model7 <- h2o.gbm(y = response, x = predictors,
		data = h2o_train2,
		distribution = "gaussian",
		n.tree = 150, shrinkage = 0.2, interaction.depth = 15)

	#### Loop over and append to data frame
		
	## Prediction
	append_results1$Hazard <- as.data.frame(h2o.predict(model6,h2o_test))$predict
	append_results2$Hazard <- as.data.frame(h2o.predict(model7,h2o_test))$predict

	# if first
	if (i == 1) {
		results <- rbind(append_results1, append_results2)
	} else
		results <- rbind(results, append_results1, append_results2)
}


# use aggregate to get mean or median for for every id
aggdata_mean <- aggregate(Hazard ~ Id, results, "mean")
aggdata_median <- aggregate(Hazard ~ Id, results, "median")

head(aggdata_mean)
mean(train$Hazard); mean(aggdata_mean$Hazard); sd(train$Hazard); sd(aggdata_mean$Hazard)

# Plot two overlaying histograms
plot1 <- density(aggdata_median$Hazard)
plot2 <- density(train$Hazard)
plot(plot1, col=rgb(0,0,1,1/4), xlim=c(0,15), main = "Hazard")
lines(plot2, col=rgb(1,0,0,1/4), xlim=c(0,15))

# Sava results
submission <- read.csv("sample_submission.csv", header = TRUE)
submission$Hazard <- exp(aggdata_median$Hazard) - 1
sub_name <- "gbm_t150_d15_rate20_log1_m20_mean"
head(submission)

write.table(submission, file = paste('results/', sub_name,'.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")

# Shut down	
h2o.shutdown(localH2O)