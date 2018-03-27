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

#select only primary and agriculture
labels$temp = lapply(labels$one_hot, function(x){
  x[c(2,3)]
  
})

labels_select = list()
for(i in 1:nrow(labels)){
if( (labels$temp[i][[1]][1] +  labels$temp[i][[1]][2] ) !=  0 ){
  labels_select[[i]] = labels[i,]
}
}


labels = rbindlist(labels_select)


labels$temp = unlist( lapply(labels$temp, function(x){
  if(x[2] == 1){
    return(1)
  }else{
    return(0)
  }
}))



saveRDS(labels, file.path('db', 'labels.rds'))