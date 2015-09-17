# Stacking ensemble / weighting
# Shuffle and split data to separate train, test and validation set

#library(auc)
library(data.table)

# Parameters
make_splits <- 2
output_dir <- "featuredate2"

# Working directory
setwd("C:/marketingdata")

# Set seed and locale for datetime
set.seed(6) # weight2: seed 17, weight3: seed 19
Sys.setlocale("LC_TIME", "English")

# Load data # improve this with: na=c("",NA,-1,9999,999))
train <- fread("C:/marketingdata/data/train.csv",
	na.strings=c("NA","N/A","", "999", "9999"),stringsAsFactors=FALSE)
test <- fread("C:/marketingdata/data/test.csv",
	na.strings=c("NA","N/A","", "999", "9999"),stringsAsFactors=FALSE)

#train.unique.count=lapply(train, function(x) length(unique(x)))
#train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
#train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
#train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

#delete_const=names(train.unique.count_1)
#delete_NA56=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145175))
#delete_NA89=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==145142))
#delete_NA918=names(which(unlist(lapply(train[,(names(train) %in% names(train.unique.count_2))], function(x) max(table(x,useNA='always'))))==144313))

#VARS to delete
#safe to remove VARS with 56, 89 and 918 NA's as they are covered by other VARS
#print(length(c(delete_const,delete_NA56,delete_NA89,delete_NA918)))

#train=train[,!(names(train) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918)),with = FALSE]
#test=test[,!(names(test) %in% c(delete_const,delete_NA56,delete_NA89,delete_NA918)),with = FALSE]

### Date and timestapm values:

datecolumns = c("VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158",
	"VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176", "VAR_0177",
	"VAR_0178", "VAR_0179","VAR_0217")

# apply for timestamp columns
gettimestamp <- function(var1, var_name = "X1") {
	gettimestamp <- strptime(var1, format="%d%b%y:%H:%M:%S")
	frame1 <- data.frame(new_wd = as.character(substr(weekdays(gettimestamp), 1, 3)), stringsAsFactors=FALSE)
	frame1$new_hour = hour(gettimestamp)
	frame1$new_month = month(gettimestamp)
	frame1$new_year = year(gettimestamp)
	frame1$numdate = as.double(as.Date(gettimestamp))
	names(frame1) <- paste(var_name,c("wd", "hour", "month", "year","date"),sep = "_")
	return(frame1)
}

getdate <- function(var1, var_name = "X1") {
	gettimestamp <- strptime(var1, format="%d%b%y:%H:%M:%S")
	frame1 <- data.frame(new_wd = as.character(substr(weekdays(gettimestamp), 1, 3)), stringsAsFactors=FALSE)
	frame1$new_month = month(gettimestamp)
	frame1$new_year = year(gettimestamp)
	frame1$numdate = as.double(as.Date(gettimestamp)) # to count difference of days
	names(frame1) <- paste(var_name,c("wd", "month", "year","date"),sep = "_")
	return(frame1)
}

train <- cbind(train,gettimestamp(train$VAR_0204, var_name = "VAR_0204"))
test <-  cbind(test,gettimestamp(test$VAR_0204, var_name = "VAR_0204"))

for (datecolumn in datecolumns) {
	train <- cbind(train, getdate(train[[datecolumn]], var_name = datecolumn))
	test <- cbind(test, getdate(test[[datecolumn]], var_name = datecolumn))
}

# Drop datecolumns and VAR_0204
train <- train[, !c(datecolumns,"VAR_0204"), with=FALSE]
test <- test[, !c(datecolumns,"VAR_0204"), with=FALSE]

# Add more features:

	# Combine hour to continent / city variable? -- dimension reduction
	# Impute missing values for 4-10 most important variables
	# New variables: Differences between time variables (date)

# Correct missing values and modify variables
feature.names <- names(train)[2:ncol(train)]
feature.names <- feature.names[-which(feature.names == "target")] # keep target

#### ADDITION: #SKIP THIS?
cat("assuming text variables
 are categorical & replacing them with numeric ids\n")

for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

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

sample0_train <- sample0[1:floor(length(sample0)*0.90)]
sample0_valid <- sample0[(floor(length(sample0)*0.90)+1):length(sample0)]
sample1_train <- sample1[1:floor(length(sample1)*0.90)]
sample1_valid <- sample1[(floor(length(sample1)*0.90)+1):length(sample1)]

# Compare this to results
table(train$X0$target)
table(train$X1$target)

# Save 2-3 datasets to output_dir

# Shuffle train to be able to split with h2o:
valid <- rbind(train$X1[sample1_valid,],train$X0[sample0_valid,])
train <-rbind(train$X1[sample1_train,],train$X0[sample0_train,])
head(train$ID)
train <- train[sample(1:dim(train)[1], dim(train)[1]),]
head(train$ID)

# Compare to previous
table(train$target) + table(valid$target)

# Write sets

write.table(test, file = paste(output_dir,'/test.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")

write.table(train, file = paste(output_dir,'/train.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
	
write.table(valid, file = paste(output_dir,'/valid.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
print("Saving datasets done")
print("done")
