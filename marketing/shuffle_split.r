# Stacking ensemble / weighting
# Shuffle and split data to separate train, test and validation set

library(data.table)
library(bit64)

# Parameters
make_splits <- 2
output_dir <- "featuredate6"

# Working directory
setwd("C:/marketingdata")

# Set seed and locale for datetime
set.seed(6) # weight2: seed 17, weight3: seed 19
Sys.setlocale("LC_TIME", "English")

# Load data # improve this with: na=c("",NA,-1,9999,999))
train <- fread("C:/marketingdata/data/train.csv",
	na.strings=c("-1","NA","N/A","", "[]", "999", "9999", "-99999"),stringsAsFactors=FALSE)
test <- fread("C:/marketingdata/data/test.csv",
	na.strings=c("-1","NA","N/A","", "[]", "999", "9999", "-99999"),stringsAsFactors=FALSE)

# Replace missing values in numeric columns also:
train[train==-99999] <- -1
train[train==-9999] <- -1
train[train==-999] <- -1
test[test==-99999] <- -1
test[test==-9999] <- -1
test[test==-999] <- -1

## Construct time variables
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

# Scaled -1 to 1 around mean. Alternatively could use (min(a)+max(a)/2 as center?
date_scale <- paste(c(datecolumns,"VAR_0204"),"_date",sep="")
for(datec in date_scale) {
	# center
	mean_date <- mean(train[[datec]], na.rm = TRUE)
	train[[datec]] <- train[[datec]] - mean_date
	test[[datec]] <- test[[datec]] - mean_date
	# scale around -1 to 1
	max_date <- max(train[[datec]], na.rm = TRUE)
	train[[datec]] <- round(train[[datec]] / max_date,7)
	test[[datec]] <- round(test[[datec]] / max_date,7)
	# Replace NA's with 0
	train[[datec]][is.na(train[[datec]])] <- 0
	test[[datec]][is.na(test[[datec]])] <- 0
}
# head(train$VAR_0073_date)

# Drop datecolumns and VAR_0204
train <- train[, !c(datecolumns,"VAR_0204"), with=FALSE]
test <- test[, !c(datecolumns,"VAR_0204"), with=FALSE]

### Remove classes with too few obs

train.unique.count=lapply(train, function(x) length(unique(x)))
train.unique.count_1=unlist(train.unique.count[unlist(train.unique.count)==1])
train.unique.count_2=unlist(train.unique.count[unlist(train.unique.count)==2])
count_3=unlist(train.unique.count[unlist(train.unique.count)==3])
count_4=unlist(train.unique.count[unlist(train.unique.count)==4])
train.unique.count_2=train.unique.count_2[-which(names(train.unique.count_2)=='target')]

delete_const <- names(train.unique.count_1)
delete_NA89 <- names(train.unique.count_2)
count_3_names <- names(count_3) # lots of useless data here..
count_4_names <- names(count_4)

# Keep with over 1000 values
keep_NOT_NA <- c("VAR_0466", "VAR_0563", "VAR_0566", "VAR_0567", "VAR_0732", "VAR_0733", "VAR_0736",
	"VAR_0737", "VAR_0739", "VAR_0740", "VAR_0741", "VAR_0924", "VAR_1162", "VAR_1163",
	"VAR_1165")
delete_NA89_MOD <- delete_NA89[-which(delete_NA89 %in% keep_NOT_NA)]

# Tarkastelua
for (vari in delete_const) {
	print(vari)
	print(table(train[[vari]]))
}

train = train[,!(names(train) %in% c(delete_const, delete_NA89_MOD)),with = FALSE]
test = test[,!(names(test) %in% c(delete_const, delete_NA89_MOD)),with = FALSE]

# Add more features:

	# Combine hour to continent / city variable? -- dimension reduction
	# Impute missing values for 4-10 most important variables
	# New variables: Differences between time variables (date)

#### ADDITION: #SKIP THIS?
cat("assuming text variables
 are categorical & replacing them with numeric ids\n")

# Convert to character variables: 
# Other factors: zip codes, classifications, etc.?
# 0241 is zip code
char_var <- c("VAR_0241") # unlist(train.unique.count[unlist(train.unique.count)<=4])
#char_var <- char_var[-which(char_var == "target")]
for (ch in char_var){
	train[[ch]] <- as.character(train[[ch]])
	test[[ch]] <-  as.character(test[[ch]])
}

# $VAR_0745: classes: -99999      0      1      2      3 : -1 as missing class
# head head(train[,800:1000,with=F]) ### CONTINUE IMPROVING THIS!
# hist(train$VAR_0745) # as numeric, this scales so that 0 = missing, 0.99... = something..

### ZIP CODES

# Strip zip code to 1 and 2 number level (3 gives county/city/what??? ?)
train$VAR_0241_zip1 <-  substr(train$VAR_0241, 1, 1)
train$VAR_0241_zip2 <-  substr(train$VAR_0241, 1, 2)
train$VAR_0241_zip3 <-  substr(train$VAR_0241, 1, 3)
test$VAR_0241_zip1 <-  substr(test$VAR_0241, 1, 1)
test$VAR_0241_zip2 <-  substr(test$VAR_0241, 1, 2)
test$VAR_0241_zip3 <-  substr(test$VAR_0241, 1, 3)

# Correct missing values and modify variables
feature_names <- names(train)[2:ncol(train)]

# Do not modify date columns
feature_names <- feature_names[-which(feature_names %in% c("target",date_scale))]

categorical <- vector()
### SCALE numeric between 1 and 0  and convert categorial to labels 1, 2, 3, ..
for (f in feature_names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
	categorical <- c(categorical,f)
  } #else {
	# Scale 0 to 1:
	#mini <- min(train[[f]], test[[f]], na.rm=TRUE)
	#train[[f]] <- train[[f]] - mini
	#test[[f]] <- test[[f]] - mini
	#maxi <- max(train[[f]], test[[f]], na.rm=TRUE)
	#train[[f]] <- train[[f]] / maxi
	#test[[f]] <- test[[f]] / maxi
	# Replace NA with 0 or -1
	#train[[f]][is.na(train[[f]])] <- 0
	#test[[f]][is.na(test[[f]])] <- 0
  #}
}

print(categorical)

#### ADDITION:
cat("replacing missing date values with 0\n") # this could introduce bias?
train[is.na(train)] <- -2 # Missing dates are 0
test[is.na(test)]   <- -2

##### Shuffle and split #####
set.seed(19) 
train <- split(train, train$target)
names(train) <- c("X0","X1")

# Stratified sampling
sample0 <- sample(dim(train$X0)[1], size = dim(train$X0)[1])
sample1 <- sample(dim(train$X1)[1], size = dim(train$X1)[1])

sample0_train <- sample0[1:floor(length(sample0)*0.80)]
sample0_valid <- sample0[(floor(length(sample0)*0.80)+1):length(sample0)]
sample1_train <- sample1[1:floor(length(sample1)*0.80)]
sample1_valid <- sample1[(floor(length(sample1)*0.80)+1):length(sample1)]

# Compare this to results
table(train$X0$target)
table(train$X1$target)

# Save 2-3 datasets to output_dir

# Shuffle train to be able to split with h2o:
valid <- rbind(train$X1[sample1_valid,],train$X0[sample0_valid,])
train <-rbind(train$X1[sample1_train,],train$X0[sample0_train,])
# Shuffle train set
head(train$ID)
train <- train[sample(1:dim(train)[1], dim(train)[1]),]
head(train$ID)
# Shuffle validation set
head(valid$ID)
valid <- valid[sample(1:dim(valid)[1], dim(valid)[1]),]
head(valid$ID)

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
