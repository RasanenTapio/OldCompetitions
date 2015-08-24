setwd("C:/mutualdata")
library(h2o)

# remember to set seed

train <- read.csv("mutualdata/train.csv", header = TRUE)
test <- read.csv("mutualdata/test.csv", header = TRUE)
results <- read.csv("sample_submission.csv", header = TRUE)

# drop id
train <- train[, -1]

for(i in 2:dim(train)[2]) {
	train[,i] <- as.factor(train[,i])
}

for(i in 2:dim(test)[2]) {
	test[,i] <- as.factor(test[,i])
}

# if sample:
otos <- sample(1:dim(train)[1], dim(train)[1]*0.1)

# Hazard or log+1 hazard?
train$Hazard <- log(train$Hazard + 1)
# kokeile 1/16 root
train$Hazard <- (train$Hazard)^(1/16)

trainsmall <- train[-otos,]
valid <- train[otos,]
# cluster
localH2O <- h2o.init(nthread=4,Xmx="10g")

h2o_train <- as.h2o(localH2O,trainsmall)
h2o_valid <- as.h2o(localH2O,valid)
h2o_test <- as.h2o(localH2O,test)
h2o_trainfull <- as.h2o(localH2O,train)

response <- 1
predictors <- 2:33

# Some models (Gradient boosted trees)

model6 <- h2o.gbm(y = response, x = predictors,
	data = h2o_train, validation=h2o_valid,
	distribution = "gaussian",
	n.tree = 700, shrinkage = 0.02, interaction.depth = 8)
	
model7 <- h2o.gbm(y = response, x = predictors,
	data = h2o_train, validation=h2o_valid,
	distribution = "gaussian",
	n.tree = 700, shrinkage = 0.02, interaction.depth = 8)
	
# 351, c(10,8,11,7), 0.02)

	# family = gaussian?
	# importance = T) ja max depth 9?

model7x <- h2o.deeplearning(y = response, x = predictors,
	data = h2o_train, validation=h2o_valid,
	epochs = 20, hidden = c(200,200,200)
	classification = FALSE)

# Deep learning model:
model_deep1 <- h2o.deeplearning(x=predictors,
                          y=response,
                          data=h2o_trainfull,
						  hidden=c(199,90),
                          epochs=50, activation="Rectifier",
                          classification=FALSE,
						  rho=0.99,
						  epsilon=1e-8)
# 3 best models without log transformation: rho, epsilin: (0.99,1.0E-6) (0.99,1.0E-8) (0.95,1.0E-6)
# 3 best models with log+1 transformation: rho, epsilin: (0.999,1.0E-8) (0.99,1.0E-8) (0.9,1.0E-6)

## Prediction
prediction1 <- as.data.frame(h2o.predict(model6,h2o_test))
prediction2 <- as.data.frame(h2o.predict(model7,h2o_test))

results$Hazard <- prediction2$predict

head(results)
mean(train$Hazard); mean(results$Hazard); sd(train$Hazard); sd(results$Hazard)

# Plot two overlaying histograms
plot1 <- density(results$Hazard)
plot2 <- density(train$Hazard)
plot(plot1, col=rgb(0,0,1,1/4), xlim=c(0,15), main = "Hazard")
lines(plot2, col=rgb(1,0,0,1/4), xlim=c(0,15))

# Sava results
results$Hazard <- (exp(prediction1$predict)-1)*0.50 + ((prediction2$predict)^16)*0.50
#results$Hazard <- (results$Hazard)^16
sub_name <- "gbm_t700_d8_rate02_log1_power16"
sub_file <- results
head(results)

write.table(sub_file, file = paste('results/', sub_name,'.csv', sep = ""), row.names = FALSE,
	quote = FALSE, col.names = TRUE, sep=",")

# h2o.shutdown(localH2O)