path <- #set path
setwd(path)

# load libraries

library(data.table)
library(lubridate)
library(lightgbm)

# load data 

train <- fread("train.csv", na.strings = c(" ","",NA))
test <- fread("test.csv",na.strings = c(" ","",NA))

# check target distribution
# In train data, only 3.6% of the ads got clicked
train[,.N/nrow(train),click]

# check time range
# The train data set contains information for 10 days from 10th Jan 2017 to 20th Jan  2017
train[,range(datetime)]

# check missing values
# siteid has around 9% missing values, browserid has around 5%, dev id has around ~14% missing values
sapply(train, function(x)(sum(is.na(x))/length(x)))

# check ID variables
# This shows train data has more number of sites, offers, categories and merchants.
for(x in c('siteid','offerid','category','merchant'))
{
  print(sprintf("There are %d unique %s in the train set and %d unique %s in the test set",train[,uniqueN(get(x))], x,test[,uniqueN(get(x))], x))
}

# Since ID variable have no ordered in them, we can't use them as is. 
# One possible option is replace them with their counts
# let's check their class first

sapply(train, class)
sapply(test, class)

# convert ID variables to their respective count
# using for-set for faster speed
cols1 <- c('siteid','offerid','category','merchant')
for(x in seq(cols))
{  
  train[,eval(cols[x]) := .N, by = eval(cols[x])]
  test[,eval(cols[x]) := .N, by=eval(cols[x])]
}

# converting other variables to count
# here we need to convert character class values to integer to show count value  under the same column
# otherwise we'll have to create new columns for these respective features and remove the old ones.
cols2 <- c('countrycode','browserid','devid')
for(x in seq(cols))
{
  train[, eval(cols[x]) := as.integer(as.factor(get(cols[x])))][,eval(cols[x]) := .N, by = eval(cols[x])]
  test[, eval(cols[x]) := as.integer(as.factor(get(cols[x])))][,eval(cols[x]) := .N, by = eval(cols[x])]
}


# Note: You might see different values such as Mozilla Firefox, Firefox, Mozilla etc. 
# You might like to club them into one value ( but look for their respective device)

  
# Extract features from datetime variable
# datetime is the timestamp at which ad got live on web
train[,datetime := as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%S")]
test[,datetime := as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%S")]

train[,tweekday := as.integer(as.factor(weekdays(datetime)))-1]
train[,thour := hour(datetime)]
train[,tminute := minute(datetime)]

test[,tweekday := as.integer(as.factor(weekdays(datetime)))-1]
test[,thour := hour(datetime)]
test[,tminute := minute(datetime)]


# Model Training
# using lightgbm for faster training than xgb
# since data is large, instead of cross validation, we'll do hold out validation
library(lightgbm)

train <- train[,.(siteid, offerid, category, merchant, countrycode, browserid, devid, tweekday, thour, tminute, click)]
test <- test[,.(siteid, offerid, category, merchant, countrycode, browserid, devid, tweekday, thour, tminute)]

trainX <- train[!folds]
valX <- train[folds]

lgb.trainX <- lgb.Dataset(as.matrix(trainX[,-c('click'),with=F]), label = trainX$click)
lgb.valX <- lgb.Dataset(as.matrix(valX[,-c('click'),with=F]), label = valX$click)

params <- list(
  objective = 'binary',
  metric = 'auc',
  feature_fraction = 0.7,
  bagging_fraction = 0.5,
  max_depth = 10
)

model <- lgb.train(params = params
                   ,data = lgb.trainX
                   ,valids = list(valid = lgb.valX)
                   ,learning_rate = 0.1
                   ,early_stopping_rounds = 40
                   ,eval_freq = 20
                   ,nrounds = 500
                   )

# get feature importance
lgb.plot.importance(tree_imp = lgb.importance(model,percentage = TRUE))

# make predictions
preds <- predict(model, data = as.matrix(test), n = model$best_iter)

# make submission
sub <- data.table(ID = test$ID, click = preds)
fwrite(sub, "lgb_starter.csv") #~ 0.6298

# What to do next ? 
# 1. Tune the parameters
# 2. Create more features
# 3. Try different models and ensemble
       
       
       
