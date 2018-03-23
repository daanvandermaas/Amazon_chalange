#create labels
classes = c("haze",    "primary" ,      "agriculture" ,  "clear",   "water" ,    "habitation",    "road",    "cultivation" ,  "slash_burn",  "cloudy",    "partly_cloudy",     "conventional_mine", "bare_ground",  "artisinal_mine",   "blooming",  "selective_logging", "blow_down")


labels = read.csv('db/train_v2.csv')

labels$tags = as.character(labels$tags)
labels$image_name = as.character(labels$image_name)

labels$classes =  lapply( labels$tags , function(x){
  
   unlist(strsplit( as.character(x), ' ')) 
  
})

labels$numbers =  lapply( labels$classes , function(x){
ret = c()
  for(i in 1:length(x)){
  ret = c(ret, which(x[i] ==classes ))    
}
  return( ret)
})


labels$one_hot = lapply(labels$numbers, function(x){
one_hot =rep(0, length(classes))
one_hot[ x ] = 1
return(one_hot)

})

saveRDS(labels, file.path('db', 'labels.rds'))