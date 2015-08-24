# You can write R code here and then click "Run" to run it on our platform

# The readr library is the best way to read and write CSV files in R
library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

# Write to the log:
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs
library(h2o)
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
train_log<-train
train_log$Hazard<-log(train_log$Hazard)

test<-as.data.frame(test)

localH2O<-h2o.init()

h2o_train<-as.h2o(localH2O,train)
h2o_train_log<-as.h2o(localH2O,train_log)
h2o_test<-as.h2o(localH2O,test)

gbm = h2o.gbm(x = 3:34, y = 2, 
              training_frame = h2o_train,
              distribution="AUTO",
              nfolds = 1,
              seed = 666,
              ntrees = 700,
              max_depth = 7,
              min_rows = 5,
              learn_rate = 0.02)

gbm_log = h2o.gbm(x = 3:34, y = 2, 
              training_frame = h2o_train_log,
              distribution="AUTO",
              nfolds = 1,
              seed = 666,
              ntrees = 700,
              max_depth = 7,
              min_rows = 5,
              learn_rate = 0.02)

pred<-as.data.frame(h2o.predict(gbm, h2o_test))
pred_log<-as.data.frame(h2o.predict(gbm_log, h2o_test))

pred_final<-0.4*pred+0.6*exp(pred_log)

write.csv(data.frame(Id=test$Id, Hazard=pred_final$predict),"R_H2O_gbm_benchmark.csv",row.names=F, quote=FALSE)
