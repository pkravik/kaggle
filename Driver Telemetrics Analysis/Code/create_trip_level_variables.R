
lagpad <- function(x, k) {
  # Function for creating lags. This lags vector x by k spots
  c(rep(NA, k), x)[1 : length(x)]
}

create_trip_level_vars <- function (DERIVED_PATH, TRIP_LEVEL_PATH, drivers_list) {
  for (driver_id in drivers_list){
    #Load the dataframe
    file_name <- paste0(DERIVED_PATH, "/", driver_id, ".rds")
    df <- readRDS(file_name)
    
    #df$lagged_change_speed <- lagpad(df$change_speed, 1)
    #use data.table for now
    dt <- data.table(df)
    
    #Should create a flag for big jumps
    
    #THIS IS SO BAD IT HURTS, flag for big jumps in speed
    dt[, is_jump := abs(change_speed)>15]
    dt[, is_jump_smooth := abs(smooth_change_speed) > 15]
    
    num_jumps = dt[, list(num_jumps = sum(is_jump, na.rm=TRUE)), by=trip_id]
    
    #Create variables with data.table. A bit unwieldge, maybe not the best way
    
    #just raw data
    # total_distance - total distance (miles)
    # total_length_sec - total length in sec
    # average_speed - average speed (mph)
    
    trip_level_dt <- dt[, list(total_distance = sum(speed, na.rm=TRUE), #total distance in miles
                               total_length_sec = .N, #total time length of the trip in seconds
                               average_speed = mean(speed, na.rm=TRUE)), #average speed. There should be one "N/A" as the first obs
                        by=trip_id]
    
    #everything, but take out the jumps
    # total_distance_no_jump - total distance
    # average_speed_no_jump - average speed (mph)
    # max_speed_no_jump - max speed (mph)
    # max_change_speed_no_jump - max change in speed (mph/s)
    # min_change_speed_no_jump - min chage in speed (max decelleration) (mph/s)
    # max_magnitude_acceleration_no_jump - max magnitude of acceleration
    # average_magnitude_acceleration_no_jump - average magnitude acceleration
    
    trip_level_dt_no_jumps <- dt[is_jump != TRUE, list(total_distance_no_jump = sum(speed, na.rm=TRUE), #total distance in miles
                                                  average_speed_no_jump = mean(speed, na.rm=TRUE), #average speed. There should be one "N/A" as the first obs
                                                  max_speed_no_jump = max(speed), #maximum speed
                                                  min_speed_no_jump = min(speed),
                                                  max_change_speed_no_jump = max(change_speed), #maximum increase in speed
                                                  min_change_speed_no_jump = min(change_speed), #max decrease in speed                 
                                                  max_magnitude_acceleration_no_jump = max(magnitude_acceleration), #maximum magnitude of acceleration
                                                  average_magnitude_acceleration_no_jump = mean(magnitude_acceleration, na.rm=TRUE)), #average magnitude acceleration
                                 by=trip_id]
    
    #Analysis of behavior when travelling above 60 mph
    # total_distance_above_60 - total distance
    # total_length_sec_above_60 - total length in sec
    # average_speed_above_60 - average speed (mph)
    
    trip_level_above_60 <- dt[speed>60 & is_jump!= TRUE, list(total_distance_above_60 = sum(speed, na.rm=TRUE), #total distance in miles if going above 60mph
                                             total_length_sec_above_60 = .N, #total time going above 60 mph
                                             average_speed_above_60 = mean(speed, na.rm=TRUE)), #average speed if going above 60,
                                                  by=trip_id]
    
    #Analysis of behavior when travelling between 40 and 60 mph
    # total_distance_40_60 - total distance
    # total_length_sec_bt_40_60 - total length in sec
    # average_speed_bt_40_60 - average speed (mph)
    trip_level_between_40_60 <- dt[speed <=60 & speed>=40 & is_jump!= TRUE, list(total_distance_bt_40_60 = sum(speed, na.rm=TRUE), #total distance in miles if going above 60mph
                                                             total_length_sec_bt_40_60 = .N, #total time going above 60 mph
                                                             average_speed_bt_40_60 = mean(speed, na.rm=TRUE)), #average speed if going above 60,
                                                  by=trip_id]
    
    #Analysis of behavior when travelling between 20 and 40 mph
    # total_distance_bt_20_40 - total distance
    # total_length_sec_bt_20_40 - total length in sec
    # average_speed_bt_20_40 - average speed (mph)
    
    trip_level_between_20_40 <- dt[speed <=40 & speed>=20 & is_jump!= TRUE, list(total_distance_bt_20_40 = sum(speed, na.rm=TRUE), #total distance in miles if going above 60mph
                                                                                 total_length_sec_bt_20_40 = .N, #total time going above 60 mph
                                                                                 average_speed_bt_20_40 = mean(speed, na.rm=TRUE)), #average speed if going above 60,
                                                  by=trip_id]
    
    #analysis of behavior when travelling between 10 and 20 mph
    # total_distance_bt_10_20 - total distance
    # total_length_sec_bt_10_20 - total length in sec
    # average_speed_bt_10_20 - average speed (mph)
    
    trip_level_between_10_20 <- dt[speed <= 20 & speed>= 10 & is_jump!= TRUE, list(total_distance_bt_10_20 = sum(speed, na.rm=TRUE), #total distance in miles if going above 60mph
                                                                                 total_length_sec_bt_10_20 = .N, #total time going above 60 mph
                                                                                 average_speed_bt_10_20 = mean(speed, na.rm=TRUE)), #average speed if going above 60,
                                   by=trip_id]
    
    #analysis of behavior when travelling below 10 mph
    # total_distance_below_10 - total distance
    # total_length_sec_below_10 - total length in sec
    # average_speed_below_10 - average speed (mph)
    trip_level_below_10 <- dt[speed<=10 & is_jump!= TRUE, list(total_distance_below_10 = sum(speed, na.rm=TRUE), #total distance in miles if going above 60mph
                                                                                 total_length_sec_below_10 = .N, #total time going above 60 mph
                                                                                 average_speed_below_10 = mean(speed, na.rm=TRUE)), #average speed if going above 60,
                                   by=trip_id]
    
    #calculate distance from origin
    # max_distance_origin - maximum distance from origin (Start of trip) (miles)
    # average_distance_origin - average distance from origin (start of trip) (miles)
    trip_level_distance_origin <- dt[, list(max_distance_origin = max(sqrt((x*0.000621371)^2 + (y*0.000621371)^2)),
                                            average_distance_origin = mean((x*0.000621371)^2 + (y*0.000621371)^2)), by=trip_id]
    
    
    ############################
    # Analysis of Smoothed Speed
    ############################
    
    #analyze smooth speed to determine segments of acceleration and decceleration
    
    #find local minima and maxima    
    
    #could probably clean this up and not save the intermediate stuff. Calculates the local extrema for speed
    dt[is_jump == FALSE, change_smooth_speed := c(NA, diff(smooth_speed)), by=trip_id]
    dt[is_jump == FALSE, accel_or_deccel := sign(change_smooth_speed), by=trip_id]
    
    #finds where it goes from accel to deccel, or deccel to accel
    dt[is_jump == FALSE, extrema := c(diff(accel_or_deccel), NA), by=trip_id]
    
    #create groups where each extrema bumps up to the next group
    dt[is_jump == FALSE & is.na(extrema)==FALSE, group := cumsum(abs(extrema)>1), by=trip_id]
    
    #for each segment (by trip id and between extrema), calculate total change in speed, average change, min and max
    dt[, ':=' (length_segment_sec = .N,
               length_segments_mi = sum(smooth_speed/3600),
               change_speed_segment = sum(change_smooth_speed),
               average_change_segment = mean(change_smooth_speed),
               max_change_segment = max(abs(change_smooth_speed)),
               max_speed_segment = max(smooth_speed),
               min_speed_segment = min(smooth_speed)), by=list(group, trip_id)]
    
    #If going to or coming from a stop (or low speed)
    dt[, from_stop := abs(min_speed_segment)<2]
    
    #now compute statistics for different buckets of changing speed segments (change between 5 and 10 mph, and also change more than 10 mph)
    
    #slowed down between 5 and 10 mph to stop
    condition1 <- quote(from_stop == TRUE & change_speed_segment < -5 & change_speed_segment > -10)
    suffix1 <- quote(slowing_to_stop_5_10)
    
    slowing_to_stop_smooth_5_10 <- get_average_smooth_speed_accel_max(dt, condition1, suffix1)
    
    #slow down at least 10 mph to stop
    condition2 <- quote(from_stop == TRUE & change_speed_segment < -10)
    suffix2 <- quote(slowing_to_stop_10)
    
    slowing_to_stop_smooth_10 <- get_average_smooth_speed_accel_max(dt, condition2, suffix2)
    
    #If increased by between 5 and 10 mph from stop
    condition3 <- quote(from_stop == TRUE & change_speed_segment > 5 & change_speed_segment < 10)
    suffix3 <- quote(accel_from_stop_5_10)
    
    accel_from_stop_smooth_5_10 <- get_average_smooth_speed_accel_max(dt, condition3, suffix3)
    
    #If increased by more than 10 mph from stop
    condition4 <- quote(from_stop == TRUE & change_speed_segment > 10)
    suffix4 <- quote(accel_from_stop_10)
    
    accel_from_stop_smooth_10 <- get_average_smooth_speed_accel_max(dt, condition4, suffix4)
    
    #number of seconds stopped
    trip_level_stopped <- dt[speed<1, list(time_stopped_sec = .N), by=trip_id]
    
    #Combine all of the tables into a list to later merge
    stop_tables <- list(slowing_to_stop_smooth_5_10,
                        slowing_to_stop_smooth_10,
                        accel_from_stop_smooth_5_10,
                        accel_from_stop_smooth_10,
                        trip_level_stopped)
    
    #color <- test[, abs(change_speed)>2]
    #plot(test[,smooth_speed], col=as.factor(color), type="p", pch=".", cex=5)
    #lines(test[,speed], col="blue")
    
    ############################
    # Analysis of speed/acceleration when turning
    ############################
    
    num_change_heading_spikes <- dt[is_jump == FALSE & change_heading > 90 & is.na(change_heading) == FALSE, 
                                    list(num_change_heading_spikes = .N), by=trip_id]
    
    trip_level_when_turning <- dt[is_jump == FALSE & abs(change_heading)>15 & abs(change_heading) < 90, 
                                  list(average_speed_when_turning = mean(speed),
                                       max_speed_when_turning = max(speed),
                                       min_speed_when_turning = min(speed),
                                       average_magnitude_acceleration_when_turning = mean(magnitude_acceleration),
                                       max_magnitude_acceleration_when_turning = max(magnitude_acceleration),
                                       min_magnitude_acceleration_when_turning = min(magnitude_acceleration),
                                       max_change_speed_when_turning = max(change_speed),
                                       min_change_speed_when_turning = min(change_speed),
                                       average_change_speed_when_turning = mean(change_speed)), by=trip_id ]
    
    turning_tables <- list(num_change_heading_spikes,
                           trip_level_when_turning)
    
    ############################
    # spline errors (not realy sure what they even mean)
    ############################
    
    spline_error_speed <- dt[is.na(smooth_speed_error)==FALSE, list(spline_error_speed = min(smooth_speed_error)), by=trip_id]
    spline_error_change_speed <- dt[is.na(smooth_change_speed_error)==FALSE, list(spline_error_change_speed = min(smooth_change_speed_error)), by=trip_id]
    
    spline_error_tables <- list(spline_error_speed,
                                spline_error_change_speed)
    
    ############################
    # standard deviations
    ############################
    
    dt_standard_errors <- dt[is_jump == FALSE, list(std_speed = sd(speed, na.rm=TRUE),
                                                    std_change_speed = sd(change_speed, na.rm=TRUE),
                                                    std_change_heading = sd(change_heading, na.rm=TRUE),
                                                    std_magnitude_acceleration = sd(magnitude_acceleration, na.rm=TRUE),
                                                    std_heading_acceleration = sd(heading_acceleration_relative, na.rm=TRUE),
                                                    std_smooth_speed = sd(smooth_speed, na.rm=TRUE)),
                             by=trip_id]
    
    
    
    #Found this construct here https://gist.github.com/reinholdsson/67008ee3e671ff23b568 to merge all of the data tables
    normal_tables <- list(num_jumps, 
                   trip_level_dt, 
                   trip_level_dt_no_jumps,
                   trip_level_above_60, 
                   trip_level_between_40_60, 
                   trip_level_between_20_40, 
                   trip_level_between_10_20,
                   trip_level_below_10,
                   dt_standard_errors)
    
    #tables to merge
    tables <- c(normal_tables, stop_tables, turning_tables, spline_error_tables)
    
    invisible(lapply(tables, function(i) setkey(i,trip_id)))
    merged <- Reduce(function(...) merge(..., all=T), tables)
    
    #probably need to clean up N/As here
    merged[is.na(merged)] <- 0
    
    #Should probably convert everything into % of total distance rather than absolute, or construct in addition
    #oh well
    
    file_name <- paste0(TRIP_LEVEL_PATH, "/", driver_id, ".rds")
    saveRDS(merged, file = file_name)          
  }
}

#specific variables I'm creating with the smoothed speeds
get_average_smooth_speed_accel_max <- function (dt, subset_condition, suffix) {
  varname1 <- paste("average_smooth_speed",suffix, sep="_")
  varname2 <- paste("average_smooth_change_speed",suffix, sep="_")
  varname3 <- paste("max_smooth_change_speed",suffix, sep="_")
  varname4 <- paste("max_change_speed", suffix, sep="_")
  varname5 <- paste("max_magnitude_acceleration", suffix, sep="_")
  
  tmp <- dt[eval(subset_condition),
             list(mean(smooth_speed),
                  mean(change_smooth_speed),
                  max(abs(change_smooth_speed)),
                  max(abs(change_speed)),
                  max(abs(magnitude_acceleration))),
             by=trip_id]

  setnames(tmp, c("V1", "V2", "V3", "V4", "V5"), c(varname1, varname2, varname3, varname4, varname5))
  
  return(tmp)
}

#specific variables I'm creating with non-smoothed stuff but subsetting on smooth
get_speed_accel_max <- function (dt, subset_condition, suffix) {
  varname1 <- paste("average_smooth_speed",suffix, sep="_")
  varname2 <- paste("average_smooth_change_speed",suffix, sep="_")
  varname3 <- paste("max_smooth_change_speed",suffix, sep="_")
  
  tmp <- dt[eval(subset_condition),
            list(mean(speed),
                 mean(magnitude_acceleration),
                 max(abs(change_speed))),
            by=trip_id]
  
  setnames(tmp, c("V1", "V2", "V3"), c(varname1, varname2, varname3))
  
  return(tmp)
}


# Driver-trip level variables

trip_level_vars <- function(df, stop_cutoff, speed_bottom_cutoff) {
  # Creates the number of stops, max speed, conditional average speed per trip
  
  # Create variables for number of stops  
  df$is_not_stopped <- df$speed > stop_cutoff
  df$is_not_stopped_lag <- lagpad(df$is_not_stopped, 1)
  df$stopping <- df$is_not_stopped_lag - df$is_not_stopped
  df$stops <- with(df, stopping == 1 & !is.na(stopping))
  
  # Group dataset 
  trip_id_group_df <- group_by(df, trip_id)
  
  # Create trip-level datasets
  stops_df <- summarise(trip_id_group_df, stops = sum(stops))  
  max_speed_df <- summarise(trip_id_group_df, max_speed = max(speed[!speed_spike], na.rm = TRUE))
  cond_avg_speed_df <- summarise(trip_id_group_df, cond_avg_speed = mean(speed[!speed_spike & speed > speed_bottom_cutoff], na.rm = TRUE))
  
  # Bottom code conditional average speed
  cond_avg_speed_df$cond_avg_speed[is.nan(cond_avg_speed_df$cond_avg_speed)] <- speed_bottom_cutoff
  
  # Merge all datasets
  speed_df <- merge(max_speed_df, cond_avg_speed_df, by = "trip_id", all = TRUE)
  trip_level_df <- merge(speed_df, stops_df, by = "trip_id", all = TRUE)
  
  return(trip_level_df)
}