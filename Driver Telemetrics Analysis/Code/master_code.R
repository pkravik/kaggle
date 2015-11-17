#install.packages(c("data.table", "dplyr"))
library(data.table)
library(dplyr)

username <- Sys.getenv("username")

#Define path to the code folder
CODE_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Code/R2D2/git/")

#Set the appropriate paths to allow for working locally
source(paste0(CODE_PATH, "folders.R"))

#Define functions
source(paste0(CODE_PATH, "functions.R"))

drivers_list <- sort(as.numeric(list.files(paste(RAW_DATA_PATH, sep="/"))))
#drivers_list <- drivers_list[1:5]
#drivers_list <- c(11, 12, 14)

# Append all trips together for a driver 
# (ONLY NEED TO DO THIS ONCE)
# Could be optimized using rblindlist at some point, likely much faster. Right now takes ~3-4 hrs
append_trips(RAW_DATA_PATH, DRIVER_LEVEL, drivers_list)

Rprof(paste0(PROFILE_PATH, "positionvariables.txt"))
#summaryRprof(paste0(PROFILE_PATH, "positionvariables.txt"))

#Create columns for speed and acceleration
source(paste0(CODE_PATH, "create_position_variables.R"))
create_new_position_level_variables(DRIVER_LEVEL, DERIVED_PATH, drivers_list)

Rprof(NULL)

#Create columns for speed and acceleration
source(paste0(CODE_PATH, "create_trip_level_variables.R"))

#Create the trip level variables

Rprof(paste0(PROFILE_PATH, "triplevel.txt"))
#summaryRprof(paste0(PROFILE_PATH, "triplevel.txt"))

create_trip_level_vars(DERIVED_PATH, TRIP_LEVEL_PATH, drivers_list)

Rprof(NULL)

#Generate individual predictions

#Rprof(paste0(PROFILE_PATH, "createprediction.txt"))
#summaryRprof(paste0(PROFILE_PATH, "createprediction.txt"))

generate_prediction(DERIVED_PATH, PREDICTION_PATH, drivers_list)


#Rprof(paste0(PROFILE_PATH, "combinepredictions.txt"))
#summaryRprof(paste0(PROFILE_PATH, "combinepredictions.txt"))

#Combine predictions
#combine_predictions(PREDICTION_PATH, drivers_list, "absoluteStd")
combine_predictions(PREDICTION_PATH, drivers_list, "gbmlesstrain")

#Rprof(NULL)
