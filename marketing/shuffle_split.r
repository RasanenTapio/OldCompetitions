# Stacking ensemble / weighting
# Shuffle and split data to separate train, test and validation set

#library(auc)
library(data.table)

# Parameters
make_splits <- 2
output_dir <- "weight2a"

# Working directory
setwd("C:/marketingdata")

# Set seed and locale for datetime
set.seed(6) # weight2: seed 17, weight3: seed 19
Sys.setlocale("LC_TIME", "English")

# Load data # improve this with: na=c("",NA,-1,9999,999))
train <- fread("C:/marketingdata/data/train.csv",
	na.strings=c("NA","N/A","", "999", "9999"))
test <- fread("C:/marketingdata/data/test.csv",
	na.strings=c("NA","N/A","", "999", "9999"))

### Date values: Set df$day <- weekdays(as.Date(df$date))
### cols:
datecolumns = c("VAR_0073", "VAR_0075", "VAR_0156", "VAR_0157", "VAR_0158",
	"VAR_0159", "VAR_0166", "VAR_0167", "VAR_0168", "VAR_0176", "VAR_0177",
	"VAR_0178", "VAR_0179","VAR_0217")
#timestampcolumns <- "VAR_0204"
# datecolumns <- c("VAR_0073", "VAR_0075")
# timestampcolumns <- "VAR_0204"

# apply for timestamp columns
gettimestamp <- function(var1, var_name = "X1") {
	gettimestamp <- strptime(var1, format="%d%b%y:%H:%M:%S")
	frame1 <- data.frame(new_wd = substr(weekdays(gettimestamp), 1, 3))
	frame1$new_hour = hour(gettimestamp)
	frame1$new_month = month(gettimestamp)
	frame1$new_year = year(gettimestamp)
	names(frame1) <- paste(var_name,c("wd", "hour", "month", "year"),sep = "_")
	return(frame1)
}

getdate <- function(var1, var_name = "X1") {
	gettimestamp <- strptime(var1, format="%d%b%y:%H:%M:%S")
	frame1 <- data.frame(new_wd = substr(weekdays(gettimestamp), 1, 3))
	frame1$new_month = month(gettimestamp)
	frame1$new_year = year(gettimestamp)
	names(frame1) <- paste(var_name,c("wd", "month", "year"),sep = "_")
	return(frame1)
}

train <- cbind(train,gettimestamp(train$VAR_0204, var_name = "X1"))
test <-  cbind(test,gettimestamp(test$VAR_0204, var_name = "X1"))

for (datecolumn in datecolumns) {
	test <- cbind(test,getdate(as.vector(test[,which( colnames(test)==datecolumn ), with = FALSE]), var_name = datecolumn))
	#train <- cbind(train,getdate(train[,datecolumn, with = FALSE], var_name = datecolumn))
	print(datecolumn)
}
#tail(test[,datecolumns,with=FALSE])
#tail(names(test)) -> nim
#tail(test[,nim,with=FALSE])
gettimestamp <- strptime(test[,which( colnames(test)=="VAR_0075" ), with = FALSE], format="%d%b%y:%H:%M:%S")

# combine hour to continent / city variable?

# Correct missing values and modify variables
feature.names <- names(train)[2:ncol(train)]
feature.names <- feature.names[-which(feature.names == "target")] # keep target

#### ADDITION: #SKIP THIS?
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
	sample0_train <- sample0[1:floor(length(sample0)*0.7)]
	# stack
	sample0_valid <- sample0[(floor(length(sample0)*0.7)+1):length(sample0)]
	sample1_train <- sample1[1:floor(length(sample1)*0.7)]
	# stack
	sample1_valid <- sample1[(floor(length(sample1)*0.7)+1):length(sample1)]
} else {
	sample0_train <- sample0[1:floor(length(sample0)*0.85)]
	sample0_valid <- sample0[(floor(length(sample0)*0.85)+1):length(sample0)]
	sample1_train <- sample1[1:floor(length(sample1)*0.85)]
	sample1_valid <- sample1[(floor(length(sample1)*0.85)+1):length(sample1)]
}
# Compare this to results
table(train$X0$target)
table(train$X1$target)

# Save 2-3 datasets to output_dir

valid <- rbind(train$X1[sample1_valid,],train$X0[sample0_valid,])
train <-rbind(train$X1[sample1_train,],train$X0[sample0_train,])
head(train$ID)
train <- train[sample(1:dim(train)[1], dim(train)[1]),]
head(train$ID)

# Compare to previous
table(train$target) + table(valid$target)

write.table(test, file = paste(output_dir,'/test.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")

write.table(train, file = paste(output_dir,'/train.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
	
write.table(valid, file = paste(output_dir,'/valid.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
print("done")
print("really")
