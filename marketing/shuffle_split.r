# Stacking ensemble / weighting
# Shuffle and split data to separate train, test and validation set

#library(auc)
library(data.table)

# Parameters
make_splits <- 3
splits_ratios <- c(0.70, 0.15, 0.15)
output_dir <- "stacking1"

# Working directory
setwd(C:/marketingdata)
# set seed

# Load data
train <- fread("C:/marketingdata/data/train.csv")

##### Shuffle and split #####
train <- split(train, train$target)
names(train) <- c("X0","X1")

# Stratified sampling
sample0 <- sample(dim(train$X0)[1], size = dim(train$X0)[1])
sample1 <- sample(dim(train$X1)[1], size = dim(train$X1)[1])

sample0a <- c(rep(1, length(sample0)*0.7),
	rep(2, length(sample0)*0.15),
	rep(3, length(sample0)*0.16))

sample1a <- c(rep(1, length(sample1)*0.7),
	rep(2, length(sample1)*0.15),
	rep(3, length(sample1)*0.16))
	
# check:
length(sample1); length(sample1a)
sample1a <- sample1a[1:length(sample1)]; length(sample1a)
length(sample0); length(sample0a)
sample0a <- sample0a[1:length(sample0)]; length(sample0a)

# Save 2-3 datasets to output_dir
