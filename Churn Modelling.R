
churn <- Churn_Modelling
summary(churn)
str(churn)

#removing rownumber,customerID and surname columns
churn<- churn[,-c(1:3)]

#converting Gender and Geography from character to numeric
churn$Gender <- ifelse(churn$Gender =="Female" , 1, 0)
churn$Geography <- factor(churn$Geography, levels = c("France", "Germany", "Spain"), 
                          labels = c(1,2,3))
churn$Geography <- as.numeric(churn$Geography)
str(churn)

#performing Data Exploration
install.packages(DataExplorer) 
library(DataExplorer)
plot_str(churn)
rm(list=ls())
plot_missing(churn)
churn
plot_histogram(churn)
plot_density(churn)
plot_correlation(churn, type = 'continuous')
plot_bar(churn)
create_report(churn)

#splitting data into training and validation partition
set.seed(2)
numberOfRows <- nrow(churn)
train.index <- sample(numberOfRows, numberOfRows*0.7)  
train.df <- churn[train.index,]
valid.df <- churn[-train.index,]

#performing logistic regression with all the features
LogModel <- glm(Exited ~ .,family=binomial(link="logit"),data=train.df)
print(summary(LogModel))

#performing logistic regression with the 4 most significant features
LogModel <- glm(Exited ~  Age+Gender+ IsActiveMember+ balance_group,family=binomial(link="logit"),data=train.df)

summary(LogModel)

#printing the confusion matrix
confusionMatrix(table(predict(LogModel, newdata = valid.df, 
                              type="response") >= 0.5, valid.df$Exited == 1))
log.pred <- predict(LogModel, valid.df, type = "response") 
log.pred

anova(LogModel, test="Chisq")

t.df <- data.frame("Predicted" = LogModel, "Label" = as.factor(churn$Exited))
View(t.df)

write.csv(export.df, file = "UBpropensities.csv")

pred <- prediction(t.df$Predicted, t.df$Label)
perf <- performance( pred, "tpr", "fpr" )
plot( perf )

#decision tree
library(rpart) 
library(rpart.plot)
library(caret)
library(e1071)
.ct <- rpart(Exited ~ ., data = train.df, method = "class", cp = 0, maxdepth = 4, minsplit = 20)
printcp(.ct)
prp(.ct, type = 1, extra = 2, under = FALSE, split.font = 1, varlen = -10)
ct.pred <- predict(.ct, valid.df, type = "class")
confusionMatrix(ct.pred, as.factor(valid.df$Exited))
summary(churn)

# build a deeper classification tree
max.ct <- rpart(Exited ~ ., data = train.df, method = "class", cp = 0, minsplit = 1, maxdepth = 30)

# count number of leaves
length(max.ct$frame$var[max.ct$frame$var == "<leaf>"])

# plot tree
prp(max.ct, type = 1, extra = 1, under = TRUE, split.font = 1, varlen = -10, 
    box.col=ifelse(max.ct$frame$var == "<leaf>", 'gray', 'white'))  

# classify records in the training data to show that the tree prefectly fits the training data.
# this is an example of overfitting
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, train.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(train.df$Exited))


# classify records in the validation data.
# set argument type = "class" in predict() to generate predicted class membership.
max.pred <- predict(max.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(max.pred, as.factor(valid.df$Exited))


# Create code to prune the tree
# xval refers to the number of partitions to use in rpart's built-in cross-validation
# procedure argument.  With xval = 5, bank.df is split into 5 partitions of 1000
# observations each.  A partition is selected at random to hold back for validation 
# while the remaining 4000 observations are used to build each split in the model. 
# Process is repeated for each parition and xerror is calculated as the average error across all partitions.
# complexity paramater (cp) sets the minimum reduction in complexity required for the model to continue.
# minsplit is the minimum number of observations in a node for a split to be attempted.
cv.ct <- rpart(Exited ~ ., data = churn, method = "class", 
               control = rpart.control(cp = 0.00000005, minsplit = 5, xval = 5))

# use printcp() to print the table. 
printcp(cv.ct)
prp(cv.ct, type = 1, extra = 1, split.font = 1, varlen = -10)  

#prune the tree using the lowest value for xerror
#Note: the prune function requires cp as a parameter so we need to get cp for lowest value of xerror
pruned.ct <- prune(cv.ct, 
                   cp = cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])

#get count of the number of splits
cp_df <- data.frame(pruned.ct$cptable)
max(cp_df$nsplit)

#another way to get the count of the number of splits
pruned.ct$cptable[which.max(pruned.ct$cptable[,"nsplit"]),"nsplit"]

#get count of the number of nodes
length(pruned.ct$frame$var[pruned.ct$frame$var == "<leaf>"])

#plot the best fitting tree
prp(pruned.ct, type = 1, extra = 1, split.font = 1, varlen = -10,
    box.col=ifelse(pruned.ct$frame$var == "<leaf>", 'gray', 'white'))  

prune.pred <- predict(pruned.ct, valid.df, type = "class")

# generate confusion matrix for training data
confusionMatrix(prune.pred, as.factor(valid.df$Exited))


#random forest
library(randomForest)
rf <- randomForest(as.factor(Exited) ~ ., data = train.df, 
                   ntree = 10000, mtry = 7, nodesize = 1, importance = TRUE) 
varImpPlot(rf, type = 1)
?randomForest
#create a confusion matrix
valid.df$Exited <- factor(valid.df$Exited)
rf.pred <- predict(rf, valid.df)
confusionMatrix(rf.pred, valid.df$Exited)

#boosted tree
#instead of randomly sampling with replacement (as in random forest) boosted trees assign weights to each obs in a bag such that
#obs with highest error are given the highest weights and are more likely to be sampled next time
library(adabag)
library(rpart)
library(caret)

boost <- boosting(Personal.Loan ~ ., data = train.df)
pred <- predict(boost, valid.df)
confusionMatrix(as.factor(pred$class), valid.df$Personal.Loan)




