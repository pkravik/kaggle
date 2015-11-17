install.packages("glmnet")
install.packages("gbm")
install.packages("caret", dependencies = c("Depends", "Suggests"))
library("glmnet")
library("gbm")
library("caret")
library("randomForest")


#Generate empty list to populate
num_drivers = length(drivers_list)
data_list <- vector("list", num_drivers)
i = 1

#loop through list and read all of the data
for (driver_id in drivers_list) {
  file_name <- paste0(TRIP_LEVEL_PATH, "/", driver_id, ".rds")
  data_list[[i]] <- readRDS(file_name)
  i=i+1
}

#append the data
data <- rbindlist(data_list)

data$driver <- rep(drivers_list, each=200)

num_other_driver_obs = 400
set.seed(885)
for (driver_id in drivers_list){
  
  #hard coded length of dataframe, prob bad
  driver_data <- data[driver==driver_id]
  other_drivers_data <- data[driver!=driver_id][sample(547000, num_other_driver_obs)]
  
  #Combine to make train data
  train_data <- rbind(driver_data, other_drivers_data)
  
  #make the outcome variable
  train_data[, outcome:=(driver == driver_id)]
  
  #probably a way to do this with data.table but I'm bad
  train_data$outcome[train_data$outcome==TRUE] <- 1
  
  gbm1 <- gbm(outcome~.-trip_id-driver, data=train_data, interaction.depth = 2, shrinkage=.01, n.trees=500, distribution="bernoulli")
  gbm_predictions <- predict(gbm1, driver_data, type="response", n.trees=500)
  
  #predictions <- gbm_predictions
  
  df <- data.frame(driver_trip = driver_data[, paste(driver, trip_id, sep="_")], prob=gbm_predictions)
  file_name <- paste0(PREDICTION_PATH, "/", driver_id, ".rds")
  saveRDS(df, file = file_name)
  
  #rf <- randomForest(as.factor(outcome)~ . - trip_id - driver, data=train_data, ntree=1000, mtry=2)
  #rf_predictions <- predict(rf, driver_data, type="prob", n.trees = 200)[,2]
  
  #predictions <- rf_predictions
  
#   df <- data.frame(driver_trip = driver_data[, paste(driver, trip_id, sep="_")], prob=rf_predictions)
#   file_name <- paste0(PREDICTION_PATH, "/rf_", driver_id, ".rds")
#   saveRDS(df, file = file_name)
}

#FINDING OPTIMAL MODEL

driver_id = 11

#gbm
gbmGrid <-  expand.grid(interaction.depth = c(2,5,8,10,11),
                        n.trees = c(500,1000),
                        shrinkage = c(0.01))

ctrl <- trainControl(method = "cv",
                     number = 5,
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE)

set.seed(885)
gbmFit3 <- train(as.factor(outcome) ~ .-trip_id-driver, data=train_data,
                 method = "gbm",
                 trControl = ctrl,
                 verbose = FALSE,
                 tuneGrid = gbmGrid,
                 metric = "ROC")

gbmFit3

#randrom forest
rfGrid <-  expand.grid(mtry = c(2, 5, 10, 30))


rf_model <- train(as.factor(outcome) ~ .-trip_id-driver, data=train_data,
                  method="rf",
                  trControl=ctrl,
                  tuneGrid = rfGrid,
                  metric = "ROC")

rf_model
