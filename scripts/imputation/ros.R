require(NADA)

files <- list.files(path = ros_inputs_location)

for (file in files) {
  
  # Open file
  df <- read.csv(paste(ros_inputs_location, '/' ,file, sep = ""))
  
  # Convert 1/0 to True/False
  df$Result_val_cen <- as.logical(df$Result_val_cen)
  
  ros_res = ros(df$Result_val, df$Result_val_cen, forwardT="log", reverseT="exp")
  
  png(file=paste(ros_analysis_location, '/' , file, '.png', sep = ""),
      width=600, height=350)
  plot(ros_res, sub="Lognormal transformation",plot.censored=TRUE)
  dev.off()
  
  summary(ros_res)
  median(ros_res); mean(ros_res); sd(ros_res)
  quantile(ros_res)
  
  ros_res.df <- as.data.frame(ros_res)
  
  write.csv(ros_res.df,paste(ros_outputs_location, '/' ,file, sep = ""), row.names = FALSE)
  
}
