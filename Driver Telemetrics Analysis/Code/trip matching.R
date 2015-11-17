install.packages("dtw")
library("dtw")
library(data.table)
library(dplyr)

username <- Sys.getenv("username")

#Define path to the code folder
CODE_PATH <- paste0("C:/Users/", username, "/Desktop/Kaggle/Driver Telemetrics Analysis/Code/R2D2/git/")

#Set the appropriate paths to allow for working locally
source(paste0(CODE_PATH, "folders.R"))

drivers_list <- sort(as.numeric(list.files(paste(RAW_DATA_PATH, sep="/"))))

drivers_list <- drivers_list[233:253]

#testing.
drivers_list <- drivers_list[1:270]

#core1
#drivers_list <- drivers_list[1:500]
#drivers_list <- drivers_list[268:500]

#core2
#drivers_list <- drivers_list[501:1000]
#screwed up drivers_list <- drivers_list[68:500]

#core3
#drivers_list <- drivers_list[1001:1500]
#drivers_list <- drivers_list[11:500]

#server
#drivers_list <- drivers_list[1501:2000]

#core1 run2
#drivers_list <- drivers_list[2001:2200]

#core2 run2
#drivers_list <- drivers_list[2201:2400]

#core3 run2
#drivers_list <- drivers_list[2401:2600]

#server run2
#drivers_list <- drivers_list[2401:2600]

#core4?
#drivers_list <- drivers_list[2501:2736]

matches_across_drivers <- function(drivers_list){
  #position level
  num_drivers = length(drivers_list)
  position_level_list <- vector("list", num_drivers)
  i = 1  
  
  for (driver_id in drivers_list){
    #get rotated trip level data
    position_level <- readRDS(paste0(TRIP_LEVEL_PATH, "/", driver_id, ".rds"))
    position_level <- data.table(position_level)
    position_level_list[[i]] <- position_level
    i <- i+1
  }
  
  trip_data <- rbindlist(position_level_list)
  trip_data[, driver := rep(drivers_list, each=200)]
  
  #trip level
  
  
  distance_matrix_list <- vector("list", num_drivers)
  i = 1
  
  for (driver_id in drivers_list){
    distance_matrix_list[[i]] <- data.table(readRDS(paste0(TRIP_MATCHING_PATH, "/", driver_id, ".rds")))
    i <- i+1
  }
  
  distance_data <- rbindlist(distance_matrix_list)
  
  #sum the distances
  distance_data[, sum_match_distances := V1^2+V2^2+V3^2+V4^2+V5^2]
  distance_data[, trip_id := rep(1:200,2736)]
  distance_data[, driver := rep(drivers_list, each=200)]
  
  merged <- merge(distance_data[, list(trip_id, driver, V1,V2,V3,V4,V5,sum_match_distances)], trip_data[, list(trip_id, driver, total_distance)], by=c("trip_id", "driver"))
  
  i<- 9
  tmp <- merged[abs(sum_match_distances - merged[i, sum_match_distances])/merged[2, sum_match_distances]<.1
         & abs(total_distance - merged[i, total_distance])/merged[i, total_distance]<.05]
  
  tmp[, test:= sqrt((V1-tmp[1, V1])^2+(V2-tmp[1, V2])^2+(V3-tmp[1, V3])^2+(V4-tmp[1, V4])^2+(V5-tmp[1, V5])^2), by=V1]
  
  
  
  View(merged[order(sum_match_distances)])
  
  distance_data <- as.matrix(distance_data)
  
  #convert to 200x200 comparison matrix. This matrix 
  comparisons <- matrix(ncol=2000, nrow=2000)
  
  #scaled_distance <- scale(distance)
  for (i in 1:2000){
    for (j in 1:2000){
      if ( i>j){
        comparisons[i,j] <- sum(abs(distance_data[i,]-distance_data[j,]))
        #fill in other side to make easier to work with
        comparisons[j,i] <- comparisons[i,j]
      }
    }
  }
  
  #convert to ordered list of matches
  all_matches <- sort(comparisons, index.return=TRUE)
  
  tmp <- data.table(index = all_matches$ix, distance = all_matches$x)
  tmp[index%%2000!=0, index_i := floor(index/2000)+1]
  tmp[index%%2000!=0, index_j := index%%2000]
  
  tmp[index%%2000==0, index_i:=index/2000]
  tmp[index%%2000==0, index_j:=2000]
  
  #read in distance matrix
  distance_matrix <- readRDS(paste0(TRIP_MATCHING_PATH, "/", driver_id, ".rds"))
}

plot_trips <- function(driver_id1, trip_id1, driver_id2, trip_id2){
  trip1 <- read.csv(paste0(RAW_DATA_PATH,"/",driver_id1,"/", trip_id1,".csv"))
  trip2 <- read.csv(paste0(RAW_DATA_PATH,"/",driver_id2,"/", trip_id2,".csv"))

  xaxis <- range(c(trip1$x, trip2$x))
  yaxis <- range(c(trip1$y, trip2$y))
  
  plot(trip1, type="l", col="red", xlim=xaxis, ylim=yaxis)
  lines(trip2, type="l", col="blue")  
}


for (driver_id in drivers_list){
  get_matches(driver_id)
}

#driver to analyze, read in the data
file_name <- paste0(DRIVER_LEVEL, "/", 1, ".rds")
df <- readRDS(file_name)
dt <- as.data.table(df)
trip_1<-dt[, list(distance=sqrt(x^2+y^2)), by=trip_id]
comparisons_trip_1 <- c(1,10,42,119,67)

for (driver_id in drivers_list){
  
  #driver to analyze, read in the data
  file_name <- paste0(DRIVER_LEVEL, "/", driver_id, ".rds")
  df <- readRDS(file_name)
  dt <- as.data.table(df)
  dt_distance<-dt[, list(distance=sqrt(x^2+y^2)), by=trip_id]
  
  #going to pick 5 comparisons in driver 1
  #1 is pretty good and long, 31888.55, relatively straight
  #10 is mid sized and large curvature, 1870.428
  #42 is e shaped, 5300
  #200 is pretty funky and long, 24600
  #67 12355
  
  #pick a random other driver and trip to test against

  distance <- matrix(ncol=5, nrow=200)
  
  for (i in 1:200){
    for (j in 1:5){
      distance[i,j] <- dtw(dt_distance[trip_id == i, distance], trip_1[trip_id == comparisons_trip_1[j], distance], distance.only=TRUE)$normalizedDistance
    }
  }
  
  file_name_save <- paste0(TRIP_MATCHING_PATH, "/", driver_id, ".rds")
  saveRDS(distance, file_name_save)
  
  #test <- data.table(trip = 1:200)
  #test[, c("d1","d2","d3","d4","d5") := dtw_array(dt_distance, trip, c(1,2,3,4,5)), by=trip] 
  
  #dtw_array(dt_distance, trip, c(1,2,3,4,5))
}

# convert_matches_to_predictions(drivers_list, 1:500)
# convert_matches_to_predictions(drivers_list, 501:1000)
# convert_matches_to_predictions(drivers_list, 1001:1801)
# convert_matches_to_predictions(drivers_list, 901:1800)
# convert_matches_to_predictions(drivers_list, 1801:2736)
# convert_matches_to_predictions(drivers_list, 1700:1801)
convert_matches_to_predictions <- function(drivers_list, range){
  for(driver in drivers_list[range]){
    get_matching_predictions(driver)
  }
}

combine_match_predictions <- function(drivers_list){
  num_drivers = length(drivers_list)
  prediction_list <- vector("list", num_drivers)
  i = 1
  
  for (driver_id in drivers_list){
    prediction <- readRDS(paste0(TRIP_MATCHING_PATH, "/Predictions/", driver_id, ".rds"))
    prediction[, driver:=driver_id]
    prediction_list[[i]] <- prediction
    #prediction_list[[i]] <- prediction[, list(driver_trip = paste(driver_id, trip_id, sep="_"), prob=prob)]
    i = i+1
  }
  
  final <- rbindlist(prediction_list)
  by_driver <- final[distance != Inf, list(number = .N, mean_distance = mean(distance), min_distance = min(distance), max_distance=max(distance)), by=driver]
  
  ##Merging with trip matching
  distance_data <- data[, list(trip_id = trip_id, driver = driver, total_distance = total_distance)]
  #merge with trip matching
  final_with_data <- merge(final, distance_data, by=c("trip_id", "driver"))
  
  final_with_data<- final_with_data[, list(adjusted_prob =prob, distance=distance, total_distance = total_distance, driver=driver, trip_id=trip_id)]
  #final_with_data[distance > 100, adjusted_prob:=0]
  #final_with_data[total_distance < 500, adjusted_prob:=0]
  
  final_with_data[distance <=10, adjusted_prob := 1]
  final_with_data[ distance > 10 & distance <=25, adjusted_prob := 0.9]
  final_with_data[ distance > 25 & distance <=50, adjusted_prob := 0.8]
  final_with_data[ distance > 50 & distance <=100, adjusted_prob := 0.5]
  
  final_with_data[distance > 100, adjusted_prob:=0]
  final_with_data[total_distance < 500, adjusted_prob:=0]
  
  final_with_data[, mean(adjusted_prob)]
  final_with_data[adjusted_prob==1, .N]
  
  final_with_data[, adjusted_distance:= distance]
  final_with_data[adjusted_distance == Inf, adjusted_distance:= 9999]
  final_with_data[, adjusted_prob:= 1-(adjusted_distance - min(adjusted_distance))/(max(adjusted_distance) - min(adjusted_distance))]
  
  trip_matching_prediction <- final_with_data[, list(driver_trip = paste(driver, trip_id, sep="_"), prob=adjusted_prob)]
  
  #Stats:
  #distance < 10 : 46580  (8.5%)
  #distance < 50 : 159272 (29.1%)
  #distance < 100: 243378 (44.5%)
  #distance < 300: 351916 (64.3%)
  #distance < 1000:396906 (72.5%) 
  #distance != Inf:402060 (73.3%)
  #Lets try if distance <100
  
  #with a cutoff of 50, only score 0.61670
  #with a cutoff of 100, only score 0.62498
  
  #lets just try continuous distance now hopefully there is enough precision. Only .67371
  
  
  write.csv(prediction, paste0(PREDICTION_PATH, "/final predictions/", "trip_matching_continuous_no_short", ".csv"), row.names=FALSE)
}

incomplete_matching_predictions <- function(){
  matches <- list.files(paste(TRIP_MATCHING_PATH, "Matches", sep="/"))
  sub("\\.(rds)", "","test.rds")
  matches <- as.numeric(sub("\\.(rds)", "",matches))
  
  for(match in matches){
    get_matching_predictions(match)
  }
  
  num_drivers = length(drivers_list)
  prediction_list <- vector("list", num_drivers)
  i = 1
  
  for (driver_id in drivers_list){
    if(driver_id %in% matches){
      prediction <- readRDS(paste0("C:/Users/P_Kravik/Desktop/Kaggle/Driver Telemetrics Analysis/Data/trip matching/Predictions/", driver_id, ".rds"))
    } else {
      prediction <- data.table(trip_id=1:200, prob=0.5)
    }
    
    prediction_list[[i]] <- prediction[, list(driver_trip = paste(driver_id, trip_id, sep="_"), prob=prob)]
    i = i+1
  }
  
  final <- rbindlist(prediction_list)
  
  write.csv(final, paste0(PREDICTION_PATH, "/final predictions/", "trip_matching_test_552", ".csv"), row.names=FALSE)
}

dtw_array <- function(dt_distance, index_i, array_j){
  trip_to_test<- dt_distance[trip_id == index_i, distance]
  trip_array<- sapply(array_j, function(x, trip_level) {return(trip_level[trip_id == x, distance])}, dt_distance)
  
  distance_array <- lapply(trip_array, function(x, trip_to_test) { return(dtw(x, trip_to_test, distance.only=TRUE)$normalizedDistance)},trip_to_test)
  
  #major_matches[, list(distance=calculate_dtw(get(trip_level), index_i, index_j))]
  #alignment <- dtw(x, y, distance.only=TRUE)
  #alignment_distance <- alignment$normalizedDistance
  
  return(distance_array)
}

get_matches <- function(driver_id){
  
  #get rotated trip level data
  trip_level <- readRDS(paste0(DERIVED_PATH, "/", driver_id, ".rds"))
  trip_level <- data.table(trip_level)
  rotated_trip_level <- rotate_trips(trip_level)
  
  #read in distance matrix
  distance_matrix <- readRDS(paste0(TRIP_MATCHING_PATH, "/", driver_id, ".rds"))
  
  #convert to 200x200 comparison matrix. This matrix 
  comparisons <- matrix(ncol=200, nrow=200)
  
  #scaled_distance <- scale(distance)
  for (i in 1:200){
    for (j in 1:200){
      if ( i>=j){
        comparisons[i,j] <- sum(abs(distance_matrix[i,]-distance_matrix[j,]))
        #fill in other side to make easier to work with
        comparisons[j,i] <- comparisons[i,j]
      }
    }
  }
  
  #convert to ordered list of matches
  all_matches <- sort(comparisons, index.return=TRUE)
  
  tmp <- data.table(index = all_matches$ix, distance = all_matches$x)
  tmp[index%%200!=0, index_i := floor(index/200)+1]
  tmp[index%%200!=0, index_j := index%%200]
  
  tmp[index%%200==0, index_i:=index/200]
  tmp[index%%200==0, index_j:=200]
  
  matches_with_distance <- tmp[index_i>index_j]
  
  #keep the first 100?
  major_matches <- matches_with_distance[1:750]
  
  #calculate actual dtw for the preliminary matches on rotated
  
  #calculate dtw distance of rotated trips
  major_matches[, distance_dtw_rotate:=calculate_dtw(rotated_trip_level, index_i, index_j), by=list(index_i, index_j)]
  
  #calculate total distance of the trips
  major_matches[, total_distance_i:=calculate_total_trip_length(trip_level, index_i), by=list(index_i)]
  major_matches[, total_distance_j:=calculate_total_trip_length(trip_level, index_j), by=list(index_j)]
  
  #calculate difference in total distance, in absolute and percent
  major_matches[, difference_in_distance:= abs(total_distance_i - total_distance_j)]
  major_matches[, pct_difference_in_distance:= 200*difference_in_distance/(total_distance_i + total_distance_j)]
  
  #order by rotated trip dtw distance?
  major_matches <- major_matches[order(distance_dtw_rotate)]
  
  #only grab those which have distances close to each other
  #maybe_bets <- major_matches[pct_difference_in_distance < 20]
  #View(maybe_bets)
  
  saveRDS(major_matches, paste0(TRIP_MATCHING_PATH, "/Matches/", driver_id, ".rds"))
  
  #calculate cumulative number of trips matched
  
  #length(unique(unlist(maybe_bets[1:200, list(index_i, index_j)]), use.names=FALSE))
  
#   #for major matches (ignoring distance)
#   tmp <- data.table(match=1:2000)
#   tmp[, cum := length(unique(unlist(major_matches[1:match, list(index_i, index_j)]), use.names=FALSE)), by=match]
#   
#   plot(major_matches[, distance_dtw_rotate], type="l", col="red")
#   par(new=TRUE)
#   plot(tmp[, cum], type="l", col="blue", axes=FALSE, ylab="")
#   axis(side=4)
#   
#   #for maybe bets (including distance difference threshold)
#   #for major matches (ignoring distance)
#   tmp2 <- data.table(match=1:1231)
#   tmp2[, cum := length(unique(unlist(maybe_bets[1:match, list(index_i, index_j)]), use.names=FALSE)), by=match]
#   
#   plot(maybe_bets[, distance_dtw_rotate], type="l", col="red")
#   par(new=TRUE)
#   plot(tmp2[, cum], type="l", col="blue", axes=FALSE, ylab="")
#   axis(side=4)
#   
#   #extract the actual matches (and try to assign some ordering)
#   maybe_results <- convert_pairings_to_orderings(maybe_best)
#   maybe_results <- maybe_results[order(distance)]
#   maybe_results[matchee!=Inf, create_plots(rotated_trip_level, trip_id, matchee, .I), by=trip_id]
}

get_matching_predictions <- function(driver_id){
  matches <- readRDS(paste0(TRIP_MATCHING_PATH, "/Matches/", driver_id, ".rds"))
  maybe_matches <- matches[pct_difference_in_distance < 20]
  maybe_results <- convert_pairings_to_orderings(maybe_matches)
  
#   num_matched <- maybe_results[matchee != Inf, .N]
#   num_great_match <- maybe_results[distance < 25, .N]
#   print(paste0("driver ", driver_id, "; matches: ",num_matched, ", great matches: ",num_great_match ))
#   
#   num_incredible_match <- maybe_results[distance < 10, .N]
#   num_great_match <- maybe_results[distance < 25, .N]
#   num_good_match <- maybe_results[distance < 50, .N]
#   num_okay_match <- maybe_results[distance < 100, .N]
#   num_meh_match <- maybe_results[distance < 200, .N]
#   num_bleh_match <- maybe_results[distance < 500, .N]
  
  maybe_results[, prob := convert_distance_to_score(distance), by=trip_id]
  
  trip_matching_prediction  <- maybe_results[, list(trip_id, prob, distance)]
  
  saveRDS(trip_matching_prediction, paste0("C:/Users/P_Kravik/Desktop/Kaggle/Driver Telemetrics Analysis/Data/trip matching/Predictions/", driver_id, ".rds"))
  
  #get rotated trip level data
#   trip_level <- readRDS(paste0(DERIVED_PATH, "/", driver_id, ".rds"))
#   trip_level <- data.table(trip_level)
#   rotated_trip_level <- rotate_trips(trip_level)
#   maybe_results[matchee!=Inf, create_plots(rotated_trip_level, trip_id, matchee, distance), by=trip_id]
  #plot results?
  
  
}

convert_distance_to_score <- function(distance){
  if(distance == Inf){
    return(0)
  }
  
  return(1)
  
  if (distance < 10){
    return(1)
  }
  
  if (distance < 25){
    return(0.9)
  }
  
  if (distance < 50){
    return(0.8)
  }

  if (distance < 100){
    return(0.7)
  }
  
  if (distance < 200){
    return(0.6)
  }
  
  if (distance < 500){
    return(0.5)
  }
  
  return(0.2)
}

convert_pairings_to_orderings <- function(maybe_best){
  #calculate the first appearance of each trip and the smallest distance?
  pairings <- data.table(trip_id = 1:200)
  
  results <- pairings[, get_first_pair(trip_id, maybe_best), by=trip_id]
  return(results)
}

get_first_pair <- function(trip_id, maybe_best){
  pairs <- maybe_best[index_i == trip_id | index_j == trip_id]
  if (nrow(pairs) == 0){
    return(data.table(matchee=Inf, distance=Inf, num_pairs = Inf))
  }
  
  num_pairs <- as.double(pairs[, .N])
  first_pair <- pairs[, .SD[which.min(distance_dtw_rotate)]]
  if(trip_id == first_pair[, index_i]){
    return(first_pair[, list(matchee = index_j, distance = distance_dtw_rotate, num_pairs = num_pairs)])
  } else {
    return(first_pair[, list(matchee = index_i, distance = distance_dtw_rotate, num_pairs = num_pairs)])
  }
}

create_plots <- function(trip_level, index_i, index_j, distance){
  
  dir.create(paste0(PLOT_PATH, "/Matches/", driver_id, "/"))
  
  xaxis <- range(trip_level[trip_id == index_i | trip_id == index_j, x])
  yaxis <- range(trip_level[trip_id == index_i | trip_id == index_j, y])
  
  #jpeg_file <- paste(PLOT_PATH, "/", 1, "/Matches/", "match_", index_i,"_",index_j, ".jpeg", sep = "")
  jpeg_file <- paste(PLOT_PATH, "/Matches/", driver_id, "/", "match_",index_i, ".jpeg", sep = "")
    
  #Generate the plot and save it
  jpeg(file = jpeg_file, width=800, height=800, quality=100, pointsize=20)
  plot(trip_level[trip_id==index_i,x], trip_level[trip_id==index_i,y], type="l", col="blue", xlim = xaxis, ylim = yaxis)
  lines(trip_level[trip_id==index_j,x], trip_level[trip_id==index_j,y], type="l", col="red")
  title(paste0(index_i, ", ", index_j, ", distance: ", distance))
  dev.off()
}

calculate_total_trip_length <- function(trip_level, trip) {
  return(trip_level[trip_id == trip, sum(speed, na.rm=TRUE)])
}

calculate_total_change_heading <- function(trip_level, trip) {
  return(trip_level[trip_id == trip, sum(change_heading, na.rm=TRUE)])
}


calculate_dtw <- function(trip_level, index_i, index_j) {
  
  t1<- trip_level[trip_id == index_i, list(x,y)]
  t2<- trip_level[trip_id == index_j, list(x,y)]
  
  #major_matches[, list(distance=calculate_dtw(get(trip_level), index_i, index_j))]
  alignment <- dtw(t1, t2, distance.only=TRUE)
  alignment_distance <- alignment$normalizedDistance
  
  return(alignment_distance)
}

calculate_dtw_distance <- function(trip_level, index_i, index_j) {
  
  x<- trip_level[trip_id == index_i, sqrt(x^2+y^2)]
  y<- trip_level[trip_id == index_j, sqrt(x^2+y^2)]
  
  #major_matches[, list(distance=calculate_dtw(get(trip_level), index_i, index_j))]
  alignment <- dtw(x, y, distance.only=TRUE)
  alignment_distance <- alignment$normalizedDistance
  
  return(alignment_distance)
}

rotate_trips <- function(trip_level){
  #rotate the last point to be on the x axis
  
  rotate_trips <- trip_level[ , list(trip_id, x, y, speed)]
  
  rotate_trips[, ":=" (last_x = tail(x, 1), last_y = tail(y,1)), by=trip_id]
  
  rotate_trips[ , angle := 180/pi * atan(last_y/last_x)]
  
  #change x positive, change y negative, add 360
  rotate_trips[last_x>=0 & last_y<0, angle:=angle+360]
  
  #change x negative, change y positive, add 180
  rotate_trips[last_x<0 & last_y>=0, angle := angle + 180]
  
  #both change negative, add 270
  rotate_trips[last_x<0 & last_y<0, angle := angle + 180]
  
  rotate_trips[, angle:=(360-angle)/180*pi]
  
  #fill in any NaN (end at the origin) with rotation of 0
  rotate_trips[is.nan(angle)==TRUE, angle:=0]
  
  #compute rotated x and y.
  rotate_trips[, ":=" (rotate_x = x*cos(angle)-y*sin(angle),
                       rotate_y = x*sin(angle)+y*cos(angle))]
  
  #rotate_trips[, positive_y := rotate_y > 0]
  rotate_trips[, pct_pos_y:=sum(rotate_y>=0 & speed>1,na.rm=TRUE)/sum(speed>1,na.rm=TRUE), by=trip_id]
  rotate_trips[, pct_pos_x:=sum(rotate_x>=0 & speed>1,na.rm=TRUE)/sum(speed>1,na.rm=TRUE), by=trip_id]
  
  #by distance travelled?
  
  #if not majority positive, flip
  rotate_trips[pct_pos_y < 0.5, rotate_y := -1*rotate_y]
  rotate_trips[pct_pos_x < 0.5, rotate_x := -1*rotate_x]
  
  rotate_trips[, ":=" (x = rotate_x, y=rotate_y)]
  
  #rotate_trips[, new_pct_pos:=sum(rotate_y>=0)/length(rotate_y), by=trip_id]
  
  return(rotate_trips)
  
}