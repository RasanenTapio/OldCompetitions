# Code to combine best submission with results from new models

# Libraries and working folder for R
setwd("C:/eegdata")
library(data.table)

# Update best submission with part of new model:
best <- fread("svm_ensemble_test2.csv", header = TRUE)

# New models
model1 <- fread("svm_ensemble_test3.csv", header = TRUE)
model2 <- fread("svm_ensemble_test4.csv", header = TRUE)

# Tested with subjects 2 and 4:
model0_num1 <- grep('^subj2_*', best$id, perl=T) 
model0_num2 <- grep('^subj4_*', best$id, perl=T) 
model1_num1 <- grep('^subj2_*', model1$id, perl=T)
model1_num2 <- grep('^subj4_*', model1$id, perl=T)
model2_num1 <- grep('^subj2_*', model2$id, perl=T)
model2_num2 <- grep('^subj4_*', model2$id, perl=T)

# Create 2 new submissions:
best_new1 <- best
best_new1[model0_num1,] <- model1[model1_num1,]
best_new1[model0_num2,] <- model1[model1_num2,]

best_new2 <- best
best_new2[model0_num1,] <- model2[model2_num1,]
best_new2[model0_num2,] <- model2[model2_num2,]

# Write files:
write.table(best_new1, file = paste('results/enseble_test3_2_12.csv', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
	
	
write.table(best_new2, file = paste('results/enseble_test4_2_12', sep = ""),
	row.names = FALSE, quote = FALSE, col.names = TRUE, sep=",")
