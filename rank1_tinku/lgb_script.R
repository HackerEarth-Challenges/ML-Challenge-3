path <- "set_path"
setwd(path)

# load libraries

library(data.table)
library(lubridate)
library(lightgbm)
library(dummies)
library(Matrix)
library(xgboost)

# load data 

train <- fread("train.csv", na.strings = c(" ","",NA))
test <- fread("test.csv",na.strings = c(" ","",NA))

testid = test$ID
nrowtrain = nrow(train)
test[, click := 5]

combin = rbind(train, test)
rm(train, test)
combin[,ID := NULL]
combin[,countrycode := as.integer(factor(countrycode))-1]

combin[,browserid := ifelse(is.na(browserid) == TRUE, "Other", combin$browserid)]
combin[,browserid := factor(browserid)]
levels(combin$browserid)
levels(combin$browserid) = c("Chrome","Edge","Firefox","Chrome","IE","IE",
                             "IE","Firefox","Firefox","Opera","Other","Safari")
combin[,browserid := as.integer(factor(browserid))-1]

combin[,devid := ifelse(is.na(devid) == TRUE, "Other", combin$devid)]
combin[,devid := as.integer(factor(devid))-1]
str(combin)

combin[,siteid := ifelse(is.na(siteid) == TRUE, -999, siteid)]

combin[,datetime := as.POSIXct(datetime, format = "%Y-%m-%d %H:%M:%S")]
combin[,tweekday := as.integer(as.factor(weekdays(datetime)))-1]
combin[,thour := hour(datetime)]

combin[,datetime := NULL]

str(combin)
combin[,siteid_count := .N, by = siteid]
combin[,offerid_count := .N, by = offerid]
combin[,category_count := .N, by = category]
combin[,merchant_count := .N, by = merchant]
combin[,siteid_offerid := .N, by = c("siteid", "offerid")]
combin[,siteid_cate_mer := .N , by = c("siteid", "category", "merchant")]

combin[,siteid_count := siteid_count/max(siteid_count)]
combin[,offerid_count := offerid_count/max(offerid_count)]
combin[,category_count := category_count/max(category_count)]
combin[,merchant_count := merchant_count/max(merchant_count)]
combin[,siteid_offerid := siteid_offerid/max(siteid_offerid)]
combin[,siteid_cate_mer := siteid_cate_mer/max(siteid_cate_mer)]


str(combin)
combin[,siteid_countrycode := .N, by = c("siteid", "countrycode")]
combin[,offerid_countrycode := .N, by = c("offerid", "countrycode")]
combin[,category_countrycode := .N, by = c("category", "countrycode")]
combin[,category_browserid := .N, by = c("category", "browserid")]

combin[,merchant_countrycode := .N, by = c("merchant", "countrycode")]
combin[,merchant_browserid := .N, by = c("merchant", "browserid")]

combin[,siteid_countrycode := siteid_countrycode/max(siteid_countrycode)]
combin[,offerid_countrycode := offerid_countrycode/max(offerid_countrycode)]
combin[,merchant_countrycode := merchant_countrycode/max(merchant_countrycode)]
combin[,category_countrycode := category_countrycode/max(category_countrycode)]
combin[,merchant_browserid := merchant_browserid/max(merchant_browserid)]
combin[,category_browserid := category_browserid/max(category_browserid)]

str(combin)
combin[,siteid_off_cat_coun := .N, by = c("siteid","offerid","category","countrycode")]
combin[,siteid_off_cat_bro := .N, by = c("siteid","offerid","category","browserid")]
combin[,siteid_off_cat_dev := .N, by = c("siteid","offerid","category","devid")]

colnames(combin)
combin[,c(5:25)] # or combin[,-c(1:4)]
str(combin)

traindata = combin[c(1:nrowtrain),]
testdata = combin[-c(1:nrowtrain),]
testdata[,click := NULL]
rm(combin)

dtrain = lgb.Dataset(data = as.matrix(traindata[,-c('click')]),label = traindata$click)

bst11 <- lightgbm(data = dtrain,
                  max_depth = 8,
                  learning_rate = 0.2,
                  nrounds = 70,
                  objective = "binary",
                  metric = "auc"
)


# imp1 =  lgb.importance(bst11, percentage = T)
# imp1
# colnames(testdata)
predd = predict(bst11, data = as.matrix(testdata))
head(predd)
sub3 <- data.table(ID = testid, click = predd)
fwrite(sub3, "tinku_preds_file.csv")
