library(tidyverse)
library(rpart)
library(rpart.plot)
library(reshape2)
library(ggplot2)
library(boot)
library(randomForest)
library(e1071)

train <- read.csv("train.csv", stringsAsFactors = F, na.strings = c("NA", ""))

test <- read.csv("test.csv", stringsAsFactors = F, na.strings = c("NA", ""))

str(train)
head(train)
head(test)

# test doesn't have variable "survived" s
test$Survived <- NA
head(test)

# both sets are messy, we clean in one df
df <- rbind(train,test)
head(df)


sapply(df, function(x) sum(is.na(x)))
table(df$Embarked)

# replace with the most common one
df$Embarked <- replace(df$Embarked, which(is.na(df$Embarked)), 'S')
# or we can try to predict the fare by the given fare

df <- df %>% mutate(Fare = ifelse(is.na(Fare), mean(df$Fare, na.rm=TRUE), Fare))
sum(is.na(df$Fare))

# we can make a new variable with family size, +1 include passenger
df$Fsize <- df$SibSp + df$Parch + 1
str(df$Fsize)
# we can always use the mean to fill all the missing data in age, but here I'm using the rpart
df$Fsize <- as.numeric(df$Fsize)
# make a stack plot


# make factors
to_fac <- c('Pclass','Sex','Embarked','Fsize')
df[to_fac] <- lapply(df[to_fac], factor)

set.seed(100)

tree = rpart(Age ~ Pclass + Sex + Embarked + Fsize, data = df)
summary(tree)
prp(tree)
predict_tree = predict(tree, data = df)
predict_tree

df$Age[is.na(df$Age)] <- predict(tree,newdata=df[is.na(df$Age),])
sum(is.na(df$Age))

hist(df$Age, freq=F, main='Age Distribution', col='lightblue',xlim = c(0,80))

colnames(df)


ggplot(df[!is.na(df$Survived),], aes(x = Sex, fill = Survived)) +
  geom_bar(stat='count', position='stack') +
  labs(x = 'Gender') + theme_grey()


ggplot(df[!is.na(df$Survived),], 
       aes(Age)) + geom_density(alpha=0.5, aes(fill=factor(Survived))) + 
  labs(title="Age with Survival rate")

# kids survive more
# we can create a new variable
df$Kid[df$Age < 18] <- 0 # kid = 0
df$Kid[df$Age >= 18] <- 1 # adult = 1

df$Kid <- as.numeric(df$Kid)

table(df$Kid, df$Survived)


# now split back
train <- df[1:891,]
test <- df[892:1309,]
str(train)
head(train)

hdata <- train[,c(2,3,5,6,7,8,12,13,14)]
head(hdata)

hdata$Pclass <- as.numeric(hdata$Pclass)
str(hdata)
hdata$Fsize <-as.numeric(hdata$Fsize)
hdata$Embarked <- as.numeric(hdata$Embarked)
hdata$Sex <- as.numeric(hdata$Sex)
str(hdata)

cormat <- round(cor(hdata),2)
melted_cormat <- melt(cormat)


cormat
sum(is.na(hdata))
head(melted_cormat)

ggplot(data = melted_cormat, 
       aes(x=Var1, y=Var2,fill=value)) + geom_tile()+
  geom_text(aes(Var2, Var1, label = value),
            color = "white", size = 4)



# split the train again for cross validation
sample <- sample(c(TRUE, FALSE), nrow(train), replace=TRUE, prob=c(0.8,0.2))
train1  <- train[sample, ]
test1   <- train[!sample, ]
# start with the simplest model

set.seed(110)

glm.fit = glm(Survived ~ Pclass + Age + Sex + Fsize + Kid + Embarked, data = train1, family=binomial)
summary(glm.fit)
summary(glm.fit)$coef[,4]

glm.prediction <- predict(glm.fit, newdata=test1, type='response')
glm.prediction <- ifelse(glm.prediction >= 0.5, 1, 0)

table(test1$Survived,glm.prediction)
#glm.prediction
#0  1
#0 89 14
#1 14 53
#not bad

# k fold

set.seed(100)

rf_model <- randomForest(factor(Survived) ~ Pclass + Age + Sex + Fsize + Kid + Embarked, data = train1)

str(train1)
sum(is.na(train1))

plot(rf_model, ylim=c(0,0.4))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)
# overall error rate lower than 0.2
# much more accurate to predict dead than survived

# try to improve accuracy by ajusting variables

rf_impor<- importance(rf_model)
rf_var_impor <- data.frame(Variables = row.names(rf_impor), 
                            Importance = round(rf_impor[ ,'MeanDecreaseGini'],2))

rf_rankImpor <- rf_var_impor %>% mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rf_rankImpor, aes(x = reorder(Variables, rf_impor), 
                           y = Importance, fill = rf_impor)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = "white") + labs(x = 'Variables')
# Sex, Age, Pclass, Fsize

rf.prediction <- predict(rf_model, test1)
table(test1$Survived,rf.prediction)
sum(test1$Survived==rf.prediction) / nrow(test1)
0.8
# rf.prediction
# 0  1
# 0 91 12
# 1 22 45


# for the test set
# prediction <- predict(rf_model, test)

train2 <- train1[,c("Survived","Sex","Age","Pclass","Fsize")]
test2 <- test1[,c("Survived","Sex","Age","Pclass","Fsize")]
train2$Survived <- factor(train2$Survived)
test2$Survived <- factor(test2$Survived)

# svm linear
svmfit=svm(Survived~. , data=train2, kernel="linear", cost=10,scale=FALSE)
summary(svmfit)

set.seed (1)
tuned_linear <- tune.svm(Survived ~., data = train2, kernel = "linear", cost = c(0.01,0.1,0.2,0.5,0.8,1,2,2.5,10),gamma=c(0.1,0.5,1,2,5))
summary(tuned_linear)

svm_linear <- svm(Survived~. , data=train2, kernel="linear", cost=1, gamma = 0.1, scale=FALSE)

svmpred_linear <- predict(svm_linear, test2)
head(svmpred_linear)
table(test2$Survived,svmpred_linear)
sum(test2$Survived==svmpred_linear) / nrow(test2)
#0.829
# we know it's a classification problem so let's use radial kernel
tuned_radial <- tune.svm(Survived ~., data = train2, kernel = "radial", cost = c(0.01,0.1,0.2,0.5,0.8,1,2,2.5,10),gamma=c(0.1,0.5,1,2,5))
summary(tuned_radial)

svm_radial <- svm(Survived~. , data=train2, kernel="radial", cost=1, gamma = 0.5, scale=FALSE)

svmpred_radial <- predict(svm_radial, test2)
sum(test2$Survived==svmpred_radial) / nrow(test2)
# 0.794
# why :(


