combine_predictions <- function(PREDICTION_PATH, drivers_list, prediction_name){
  # This function combines all of the individual predictions for each driver and outputs a csv to submit
  # args:
  #   PREDICTION_PATH: path to folder to save predictions. SAves in ./final predictions/name.csv
  #   drivers_list: list of drivers (integers) to iterate over
  #   prediction_name: string, name of file to save
  
  #Generate empty list to populate
  num_drivers = length(drivers_list)
  prediction_list1 <- vector("list", num_drivers)
  prediction_list2 <- vector("list", num_drivers)
  prediction_list3 <- vector("list", num_drivers)
  prediction_list4 <- vector("list", num_drivers)
  i = 1
  
  #loop through list and fill in with read data frame
  for (driver_id in drivers_list) {
    file_name <- paste0(PREDICTION_PATH, "/", driver_id, ".rds")
    prediction_list1[[i]] <- readRDS(paste0(PREDICTION_PATH, "/2 gbm and rf/", driver_id, ".rds"))
    prediction_list2[[i]] <- readRDS(paste0(PREDICTION_PATH, "/2 gbm and rf/gbm_", driver_id, ".rds"))
    prediction_list3[[i]] <- readRDS(paste0(PREDICTION_PATH, "/2 gbm and rf/rf_", driver_id, ".rds"))
    prediction_list4[[i]] <- readRDS(paste0(PREDICTION_PATH, "/2 gbm and rf/gbm2_", driver_id, ".rds"))
    
    #Should redo how this work. should probably normalize/have driver in the generate prediction part
#     blah <- readRDS(file_name)
#     blah$driver <- driver_id
#     prediction_list[[i]] <- blah
#     
#     test <- final[ , list(average = mean(prob), 
#                           std = sd(prob), 
#                           median = median(prob),
#                           ten = quantile(prob, 0.1),
#                           ninety = quantile(prob, 0.9)), by=driver]
#     
    i = i+1
  }
  
  #Combine all of the dataframes in the list
  gbm1 <- rbindlist(prediction_list1)
  gbm2 <- rbindlist(prediction_list2)
  gbm3 <- rbindlist(prediction_list4)
  rf1 <- rbindlist(prediction_list3)
  merged1 <- merge(gbm1, gbm2, by="driver_trip")
  merged2 <- merge(merged1, gbm3, by="driver_trip")
  final <- merge(merged2, rf1)
  final[, prob.z := prob]
  final[, prob_model:= (prob.x+prob.x+2*prob.z)/4]
  
  merged2[, prob_model:= (prob.x+prob.y+prob)/3]
  final <- merged2

  final <- merge(final, trip_matching_prediction[, list(driver_trip = driver_trip, prob_match=prob)], by="driver_trip")

  final[prob_match == 1, prob:=(prob_model+prob_match)/2]
  final[prob_match == 1, .N]
  #normalize by driver, doing by mean now, maybe makes sense to do by median? Think about this more
  #normalized_final <- final[ , prob := (prob-median(prob))/sd(prob), by=driver]
  #normalized_final[, prob:= (prob - min(prob))/(max(prob)-min(prob))]
  
  #normalized_final[, driver:=NULL]
  
  #Save prediction
  #write.csv(normalized_final, paste0(PREDICTION_PATH, "/final predictions/", prediction_name, ".csv"), row.names=FALSE)
  write.csv(final[,list(driver_trip=driver_trip, prob=prob_model)], paste0(PREDICTION_PATH, "/final predictions/", prediction_name, ".csv"), row.names=FALSE)
}

generate_prediction <- function(DERIVED_PATH, PREDICTION_PATH, drivers_list){
  # This function generate a prediction for each driver
  # args:
  #   DERIVED_PATH: path to data with derived variables ./driver_id.rds
  #   PREDICTION_PATH: path to folder to save predictions ./driver_id.rds
  #   drivers_list: list of drivers(integers) to iterate over
  
  for (driver_id in drivers_list) {
    file_name <- paste0(DERIVED_PATH, "/", driver_id, ".rds")
    df <- readRDS(file_name)
    
    #use data.table for now
    dt <- data.table(df)
    
    #Collapse to the driver-trip level
    trip_level <- dt[, list(total_distance = sum(speed[!speed_spike], na.rm=TRUE),
                            average_speed = mean(speed[!speed_spike], na.rm=TRUE)
    ), 
    by=trip_id]
    
    # Merge trip level variables
    trip_level <- merge(trip_level, trip_level_vars(df, 3, 3), by="trip_id", all = TRUE)
    
    #Calculate the std of distance and no of stops for each trip by a driver
    trip_level[, std_distance := (total_distance - mean(total_distance)) / sd(total_distance)]
    trip_level[, std_speed := (average_speed - mean(average_speed) / sd(average_speed))]
    trip_level[, std_stops := (stops - mean(stops))/sd(stops)]
    #####################################
    # NEED TO FIX THIS TO DEAL WITH NA'S
    #####################################
    trip_level[, std_cond_speed := (cond_avg_speed - mean(cond_avg_speed)) / sd(cond_avg_speed)]
    trip_level[, std_max_speed := (max_speed - mean(max_speed)/ sd(max_speed))]
    
    #Create driver_trip_id
    trip_level[, driver_trip := paste(driver_id, trip_id, sep="_")]
    
    #extract just driver_trip and probability, the absolute value of std_distance
    prediction <- trip_level[, list(driver_trip, prob = -1*(abs(std_distance) + abs(std_stops) 
                                                            + abs(std_speed) + abs(std_cond_speed) 
                                                            + abs(std_max_speed)))]
    
    file_name <- paste0(PREDICTION_PATH, "/", driver_id, ".rds")
    saveRDS(prediction, file = file_name)
  }
}

append_trips <- function(RAW_DATA_PATH, DRIVER_LEVEL, drivers_list) {
  # This function iterates through a list of trivers, and appends all of the 200 trips
  # into a single dataframe and saves it as an R object at DRIVER_LEVEL/driver_id.rds
  # args:
  #     RAW_DATA_PATH: path to the raw data with ./driver_id/trip_id.csv
  #     DRIVER_LEVEL: path to folder to save R objects
  #     drivers_list: list of integers corresponding to a driver
  
  for (driver_id in drivers_list) {
    
    #Initialize empty data frame
    df_driver_id <- data.frame()
    
    for (trip_id in seq(from = 1, to = 200)) {
      #Read in the csv with driver_id and trip_id
      data_path <- paste(RAW_DATA_PATH, "/", driver_id, "/", trip_id, ".csv", sep="")
      df <- read.csv(data_path)
      
      #Create a column with trip_id and append to dataframe at the driver level
      df$trip_id <- trip_id
      df_driver_id <- rbind(df_driver_id, df)
    }
    
    #Save the driver dataframe as an R ovject
    file_name <- paste0(DRIVER_LEVEL, "/", driver_id, ".rds")
    saveRDS(df_driver_id, file = file_name) 
  }
}

create_trip_plots <- function(RAW_DATA_PATH, PLOT_PATH, driver_id) {
  # This function looks through a driver's directory and generates
  # the X,Y position plot and saves it
  # args:
  #   RAW_DATA_PATH: path to folder structures with ./driver_id/trip.id.csv
  #   PLOT_PATH: path to folder to save plots with sane structure as data
  #   driver_id: see above
  
  #Create the subdirectory for driver_id
  dir.create(file.path(PLOT_PATH, driver_id))
  
  #There are 200 trips per driver, check this
  file_list <- list.files(paste(RAW_DATA_PATH, driver_id, sep="/"))
  if (length(file_list) != 200){
    stop("There are not 200 trips")
  }
  
  for (trip_id in seq(from = 1, to = 200)){
    #Define the paths for the data and the plot to be made
    data_path <- paste(RAW_DATA_PATH, "/", driver_id, "/", trip_id, ".csv", sep="")
    jpeg_file <- paste(PLOT_PATH, "/", driver_id, "/", "trip_", trip_id, ".jpeg", sep = "")
    
    #Read the raw data
    df <- read.csv(data_path)
    
    #speed <- get_speed(df)
    #acceleration <- get_acceleration(df)
    #All the other variables here
    
    #Generate the plot and save it
    jpeg(file = jpeg_file, width=800, height=800, quality=100, pointsize=20)
    plot(df$x, df$y, type="l")
    dev.off()
  }
}

