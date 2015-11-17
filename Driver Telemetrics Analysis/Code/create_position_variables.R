
create_new_position_level_variables <- function(DRIVER_LEVEL, DERIVED_PATH, drivers_list) {
  # This function creates speed and the magnitude of acceleration
  # args:
  #   DRIVER_LEVEL: path to folder containing R objects of dataframes of all trips of a driver
  #   DERIVED_PATH: path to folder to save derived files
  #   drivers_list: list of drivers(integers) to iterate over
  
  for (driver_id in drivers_list) {
    #Read in driver's trips
    file_name <- paste0(DRIVER_LEVEL, "/", driver_id, ".rds")
    df <- readRDS(file_name)
    
    #A bit inefficient, calculating trip id different twice. Oh well. Offset 1 for trip id
    change_x <- calculate_change_coordinate(df$x, df$trip_id, 1)
    change_y <- calculate_change_coordinate(df$y, df$trip_id, 1)
    
    #convert to mph from m/s
    change_x <- change_x * 2.23694
    change_y <- change_y * 2.23694
    
    #calculate the change in the change in x (Acceleration vector), offset 2 for trip id
    change_change_x <- calculate_change_coordinate(change_x, df$trip_id, 2)
    change_change_y <- calculate_change_coordinate(change_y, df$trip_id, 2)
    
    #Calculate speed
    df$speed <- calculate_speed(change_x, change_y)
    
    #calculate heading
    heading <- calculate_heading(change_x, change_y)
    df$heading <- repeat.before(heading)
    
    #calculate change in heading
    df$change_heading <- abs(df$heading - lagpad(df$heading, 1))
    df$change_heading[df$change_heading > 250 & is.na(df$change_heading) == FALSE] <- 360 - df$change_heading[df$change_heading > 250 & is.na(df$change_heading) == FALSE] 
    
    #calculate change in speed
    df$change_speed <- calculate_change_speed(df$speed)
    
    #calculate magnitude of acceleration
    df$magnitude_acceleration <- calculate_magnitude_acceleration(change_change_x, change_change_y)
    
    #calculate heading of accleration, counter clockwise from <1,0>
    df$heading_acceleration <- calculate_acceleration_heading(change_x, change_change_x, change_y, change_change_y)
    
    #Calculate relative to the previous velocity
    df$heading_acceleration_relative <- df$heading_acceleration - lagpad(df$heading,1)
    df$heading_acceleration_relative[which(df$heading_acceleration_relative < 0)] <- 360 + df$heading_acceleration_relative[which(df$heading_acceleration_relative < 0)]
    
    #This is right now?!
    df$heading_acceleration_relative[which(df$heading_acceleration_relative>180)] <- df$heading_acceleration_relative[which(df$heading_acceleration_relative>180)] - 360
    
    dt <- data.table(df)
      
    #SMOOTHING
  
    #THIS IS ALL SO BAD I'M SORRY, SHOULD WRITE A FUNCTION
    
    #exclude jumps when smoothing? Will for now, should maybe change
    #THIS IS SO BAD IT HURTS, flag for big jumps in speed
    dt[, is_jump := abs(change_speed)>15]
    
    #Speed
    dt[is_jump == FALSE & is.na(speed) == FALSE, can_smooth_speed := IQR(speed) > 0 , by=trip_id]
    dt[is_jump == FALSE & is.na(speed) == FALSE & can_smooth_speed == TRUE, smooth_speed := smooth.spline(speed)$y, by=trip_id]
    dt[is_jump == FALSE & is.na(speed) == FALSE & can_smooth_speed == TRUE, smooth_speed_error := smooth.spline(speed)$cv.crit, by=trip_id]
    dt[, can_smooth_speed := NULL]
    
    #magnitude acceleration
    dt[is_jump == FALSE & is.na(magnitude_acceleration) == FALSE, can_smooth_magnitude_acceleration := IQR(magnitude_acceleration) > 0 , by=trip_id]
    dt[is_jump == FALSE & is.na(magnitude_acceleration) == FALSE & can_smooth_magnitude_acceleration == TRUE, smooth_magnitude_acceleration := smooth.spline(magnitude_acceleration)$y, by=trip_id]
    dt[is_jump == FALSE & is.na(magnitude_acceleration) == FALSE & can_smooth_magnitude_acceleration == TRUE, smooth_magnitude_acceleration_error := smooth.spline(magnitude_acceleration)$cv.crit, by=trip_id]
    dt[, can_smooth_magnitude_acceleration := NULL]
    
    #Change speed
    dt[is_jump == FALSE & is.na(change_speed) == FALSE, can_smooth_change_speed := IQR(change_speed) > 0 , by=trip_id]
    dt[is_jump == FALSE & is.na(change_speed) == FALSE & can_smooth_change_speed == TRUE, smooth_change_speed := smooth.spline(change_speed)$y, by=trip_id]
    dt[is_jump == FALSE & is.na(change_speed) == FALSE & can_smooth_change_speed == TRUE, smooth_change_speed_error := smooth.spline(change_speed)$cv.crit, by=trip_id]
    dt[, can_smooth_change_speed := NULL]
    
    #change heading
    dt[is_jump == FALSE & is.na(change_heading) == FALSE, can_smooth_change_heading := IQR(change_heading) > 0 , by=trip_id]
    dt[is_jump == FALSE & is.na(change_heading) == FALSE & can_smooth_change_heading == TRUE, smooth_change_heading := smooth.spline(change_heading)$y, by=trip_id]
    dt[is_jump == FALSE & is.na(change_heading) == FALSE & can_smooth_change_heading == TRUE, smooth_change_heading_error := smooth.spline(change_heading)$cv.crit, by=trip_id]
    dt[, can_smooth_change_heading := NULL]
    
    #heading acceleration relative
    dt[is_jump == FALSE & is.na(heading_acceleration_relative) == FALSE, can_smooth_heading_acceleration_relative := IQR(heading_acceleration_relative) > 0 , by=trip_id]
    dt[is_jump == FALSE & is.na(heading_acceleration_relative) == FALSE & can_smooth_heading_acceleration_relative == TRUE, smooth_heading_acceleration_relative := smooth.spline(heading_acceleration_relative)$y, by=trip_id]
    dt[is_jump == FALSE & is.na(heading_acceleration_relative) == FALSE & can_smooth_heading_acceleration_relative == TRUE, smooth_heading_acceleration_relative_error := smooth.spline(heading_acceleration_relative)$cv.crit, by=trip_id]
    dt[, can_smooth_heading_acceleration_relative := NULL]
    
    # Flag speed spikes
    #df$speed_spike <- clean_speed(df, 45)
    
    file_name <- paste0(DERIVED_PATH, "/", driver_id, ".rds")
    saveRDS(dt, file = file_name)
  }
}

#found here http://stackoverflow.com/questions/7735647/replacing-nas-with-latest-non-na-value
repeat.before <- function(x) {   # repeats the last non NA value. Keeps leading NA
  ind = which(!is.nan(x))      # get positions of nonmissing values
  if(is.nan(x[1]))             # if it begins with a missing, add the 
    ind = c(1,ind)        # first position to the indices
  rep(x[ind], times = diff(   # repeat the values at these indices
    c(ind, length(x) + 1) )) # diffing the indices + length yields how often 
  }                               # they need to be repeated

calculate_acceleration_heading <- function(change_x, change_change_x, change_y, change_change_y){
  #this function calculates the relative direction of acceleration to the previous velocity
  #e.g.
  #   t0: velocity is <3, 6>
  #   t1: velocity is <4, 5>
  # Therefore,
  #   t1: acceleration is <1,-1>
  # Calculates relative angle between <3,6> and <1,-1> in degrees, where 0 is straight, and positive values
  # are clockwise from the t0 velocity
  # This can be done with a dot product
  #   a dot b = |a||b|cos(theta)
  #   so (3*1 + 6*-1) = sqrt(45) * sqrt(2) * cos(theta)
  #   theta = acos(-2/(sqrt(45)(sqrt(2))))
  # NEVERMIND, JUST BEING DUMB AND FINDING COUNTER CLOCKWISE FROM <1,0>
  
  acceleration_heading <- 180/pi * atan(change_change_y/change_change_x)
  
  #change x positive, change y negative, add 360
  acceleration_heading[which(change_change_x>=0 & change_change_y<0)] <- acceleration_heading[which(change_change_x>=0 & change_change_y<0)] + 360
  
  #change x negative, change y positive, add 180
  acceleration_heading[which(change_change_x<0 & change_change_y>=0)] <- acceleration_heading[which(change_change_x<0 & change_change_y>=0)] + 180
  
  #both change negative, add 270
  acceleration_heading[which(change_change_x<0 & change_change_y<0)] <- acceleration_heading[which(change_change_x<0 & change_change_y<0)] + 180
  
  return(acceleration_heading)
}

calculate_change_coordinate <- function(coordinate_column, trip_id_column, offset) {
  #this function takes a variable and calculates the change, accounting for the
  #data structure and replacing any across trip comparison with NA (offset is amount to offset by with trip_id)
  
  #calculate lag variables to feed into other functions
  lag <- lagpad(coordinate_column, 1)
  
  change <- coordinate_column - lag
  
  #Correct for change looking at a different trip (At the beginning)
  #lag for trip id, use offset
  trip_id_lag <- lagpad(trip_id_column, offset)
  
  #For the initial observations which are NA (first two obs) set to trip 1
  trip_id_lag[1:offset] <- 1
  
  #booleans for whether an observation is the same trip as the previous, lagged 1 and 2
  id_change <- trip_id_column != trip_id_lag
  
  #If not the same triplagged 1, then change in X and Y is NA
  change[id_change] <- NA
  
  return(change)
}

# Creating distance, velocity, and acceleration variables

lagpad <- function(x, k) {
  # Function for creating lags. This lags vector x by k spots
  c(rep(NA, k), x)[1 : length(x)]
}

calculate_speed <- function(change_x, change_y) {
  #This function will create a list of speed
  speed <- sqrt((change_x)^2+(change_y)^2)
  
  return(speed)
}


calculate_heading <- function(change_x, change_y) {
  #this function calculates the heading of movement (change in position)
  #Will return degrees, where 0 is <1,0>, to the right, and 90 degrees in <0,1>
  #both changes are positive, no change needed
  heading <- 180/pi * atan(change_y/change_x)
  
  #change x positive, change y negative, add 360
  heading[which(change_x>=0 & change_y<0)] <- heading[which(change_x>=0 & change_y<0)] + 360
  
  #change x negative, change y positive, add 180
  heading[which(change_x<0 & change_y>=0)] <- heading[which(change_x<0 & change_y>=0)] + 180
  
  #both change negative, add 270
  heading[which(change_x<0 & change_y<0)] <- heading[which(change_x<0 & change_y<0)] + 180
  
  return(heading)
  
}

calculate_magnitude_acceleration <- function(change_change_x, change_change_y) {
  #this function calculates the magnitude of the acceleration vector
  magnitude_acceleration <- sqrt((change_change_x)^2 + (change_change_y)^2)
  
  return(magnitude_acceleration)
}

calculate_change_speed <- function(speed) {
  #This function will create a list of the magnitude of acceleration
  speed_lag <- lagpad(speed, 1)
  
  change_speed <- speed - speed_lag
  return(change_speed)
}

clean_speed <- function(df, speed_top_cutoff) {
  # Creating a flag for unrealistic spikes in speed
  speed_spike <- with(df, speed > speed_top_cutoff & !is.na(speed))
  return(speed_spike)
}