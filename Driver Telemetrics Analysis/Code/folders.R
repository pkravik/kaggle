RAW_DATA_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/drivers")
CLEAN_DATA_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/Clean")
PLOT_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/trip_plots")
DRIVER_LEVEL <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/driver level")
DERIVED_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/derived/")
PREDICTION_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/predictions")
PROFILE_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/profiling/")
CODE_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Code/R2D2/git/")
TRIP_LEVEL_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/trip level")
TRIP_MATCHING_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Data/trip matching")

dir.create(file.path(DRIVER_LEVEL), showWarnings = FALSE)
dir.create(file.path(TRIP_LEVEL_PATH), showWarnings = FALSE)

#dir.create(file.path(CLEAN_DATA_PATH), showWarnings = FALSE)

dir.create(DERIVED_PATH, showWarnings = FALSE)

dir.create(PREDICTION_PATH, showWarnings = FALSE)
dir.create(paste0(PREDICTION_PATH, "/final predictions"), showWarnings = FALSE)

dir.create(PROFILE_PATH, showWarnings = FALSE)

dir.create(file.path(TRIP_MATCHING_PATH), showWarnings = FALSE)

