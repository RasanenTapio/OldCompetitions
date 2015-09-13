# Stacking ensemble / weighting
# Shuffle and split data to separate train, test and validation set

#library(auc)
library(data.table)

# Parameters
make_splits <- 2
splits_ratios <- c(0.70, 0.15, 0.15)
output_dir <- "weight5"

# Working directory
setwd("C:/marketingdata")
# Set seed

# Load data # improve this with: na=c("",NA,-1,9999,999))
train <- fread("C:/marketingdata/data/train.csv")
test <- fread("C:/marketingdata/data/test.csv")

# Correct missing values and modify variables
feature.names <- names(train)[2:ncol(train)-1]

#### ADDITION:
cat("assuming text variables
 are categorical & replacing them with numeric ids\n")
#categorical <- vector()
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
	#categorical <- c(categorical,f)
  }
}
#categorical # print categorical variable names or numbers
#### ADDITION:
cat("replacing missing values with -1\n")
train[is.na(train)] <- -1
test[is.na(test)]   <- -1

##### Shuffle and split #####
train <- split(train, train$target)
names(train) <- c("X0","X1")

# Stratified sampling
sample0 <- sample(dim(train$X0)[1], size = dim(train$X0)[1])
sample1 <- sample(dim(train$X1)[1], size = dim(train$X1)[1])

if (make_splits == 3) {
	sample0a <- c(rep(1, length(sample0)*0.7),
		rep(2, length(sample0)*0.15),
		rep(3, length(sample0)*0.16))

	sample1a <- c(rep(1, length(sample1)*0.7),
		rep(2, length(sample1)*0.15),
		rep(3, length(sample1)*0.16))
} else {
	sample0_train <- sample0[1:floor(length(sample0)*0.8)]
	sample0_valid <- sample0[(floor(length(sample0)*0.8)+1):length(sample0)]
	sample1_train <- sample1[1:floor(length(sample1)*0.8)]
	sample1_valid <- sample1[(floor(length(sample1)*0.8)+1):length(sample1)]
}

# Save 2-3 datasets to output_dir

valid <- rbind(train$X1[sample1_valid,],train$X0[sample0_valid,])
train <-rbind(train$X1[sample1_train,],train$X0[sample0_train,])

# as.matrix to speed up!

write.table(test, file = paste(output_dir,'/test.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")

write.table(train, file = paste(output_dir,'/train.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
	
write.table(valid, file = paste(output_dir,'/valid.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
print("done")
