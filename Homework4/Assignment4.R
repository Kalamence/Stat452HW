#11. This question uses the Caravan data set.

#(a) Create a training set consisting of the first 1,000 observations,
#and a test set consisting of the remaining observations.

library("ISLR")

data(Caravan)

View(Caravan)
str(Caravan)

train=Caravan[1:1000,]
test=Caravan[1001:nrow(Caravan),]


#(b) Fit a boosting model to the training set with Purchase as the
#response and the other variables as predictors. Use 1,000 trees,
#and a shrinkage value of 0.01. Which predictors appear to be
#the most important?

library("gbm")

set.seed(69)

train$Purchase=(train$Purchase=="Yes")

boost_forest=gbm(Purchase~. -PVRAAUT -AVRAAUT, data=train, distribution="bernoulli", n.trees=1000, shrinkage=0.01)

summary(boost_forest)
head(summary(boost_forest), n=8)

#(c) Use the boosting model to predict the response on the test data.
#Predict that a person will make a purchase if the estimated probability
#of purchase is greater than 20 %. Form a confusion matrix.
#What fraction of the people predicted to make a purchase
#do in fact make one? How does this compare with the results
#obtained from applying KNN or logistic regression to this data
#set?

yhat_boost=predict(boost_forest, newdata=test, n.trees=1000, type="response")

summary(yhat_boost)

yhat_boost=(yhat_boost>0.2)

test$Purchase=(test$Purchase=="Yes")

cm=table(test$Purchase, yhat_boost)
cm

cm[2,2]/sum(cm[,2])

###################

library("class")

knn_pred=knn(train=train[,1:85], test=test[,1:85], cl=train[,86], k=3, prob=TRUE)

knn_pred

yhat_knn=attr(knn_pred, "prob")

yhat_knn=(yhat_knn<0.8)

cm_knn=table(test$Purchase, yhat_knn)
cm_knn
cm_knn[2,2]/sum(cm_knn[,2])

###################

glm_fit=glm(Purchase~., data=train, family=binomial)

glm_pred=predict(glm_fit, test, type="response")

summary(glm_pred)

glm_pred=(glm_pred>0.2)

cm_glm=table(test$Purchase, glm_pred)
cm_glm

cm_glm[2,2]/sum(cm_glm[,2])

# 7. In this problem, you will use support vector approaches in order to
# predict whether a given car gets high or low gas mileage based on the
# Auto data set.

# (a) Create a binary variable that takes on a 1 for cars with gas
# mileage above the median, and a 0 for cars with gas mileage
# below the median.

data("Auto")

Auto$y=(Auto$mpg>median(Auto$mpg))
Auto$y=as.numeric(Auto$y)

# (b) Fit a support vector classifier to the data with various values
# of cost, in order to predict whether a car gets high or low gas
# mileage. Report the cross-validation errors associated with different
# values of this parameter. Comment on your results.

library("e1071")

svm_fit=svm(y~. -mpg, type="C-classification", cost=1, data=Auto, kernal="linear")

tune_auto=tune(svm, y~. -mpg, data=Auto, kernal="linear", ranges=list(cost=c(10^{-3:3})))

summary(tune_auto)

tune_auto

# (c) Now repeat (b), this time using SVMs with radial and polynomial
# basis kernels, with different values of gamma and degree and
# cost. Comment on your results.

svm_fit_rad=svm(y~. -mpg, type="C-classification", cost=1, gamma=0.5, data=Auto, kernal="radial")

tune_auto_rad=tune(svm, y~. -mpg, data=Auto, kernal="radial", ranges=list(cost=c(10^{-3:3}), gamma=c(.5, 1:4)))

summary(tune_auto_rad)

tune_auto_rad

svm_fit_poly=svm(y~. -mpg, type="C-classification", cost=1, gamma=0.5, data=Auto, kernal="polynomial")

tune_auto_poly=tune(svm, y~. -mpg, data=Auto, kernal="polynomial", ranges=list(cost=c(10^{-3:3}), gamma=c(.5, 1:4)))

summary(tune_auto_poly)$performances

tune_auto_poly

# (d) Make some plots to back up your assertions in (b) and (c).

asdf=tune_auto$performances
asdf$cost=as.factor(asdf$cost)

plot(asdf$cost, asdf$error, xlab="cost", ylab="error")

##################

library("ggplot2")

meow=tune_auto_poly$performances

str(meow)
meow$cost=as.factor(meow$cost)
meow$gamma=as.factor(meow$gamma)
summary(meow$error)

p <- ggplot(data=meow, aes(cost, gamma)) + geom_tile(aes(fill = error), 
    colour = "black") + scale_fill_gradient(low = "white", high = "red")
p

###################

merp=tune_auto_rad$performances

str(merp)
merp$cost=as.factor(merp$cost)
merp$gamma=as.factor(merp$gamma)
summary(merp$error)

p <- ggplot(data=merp, aes(cost, gamma)) + geom_tile(aes(fill = error), 
    colour = "black") + scale_fill_gradient(low = "white", high = "red")
p