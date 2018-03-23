#create labels
labels = read.csv('db/train_v2.csv')

labels$tags = as.character(labels$tags)
labels$image_name = as.character(labels$image_name)

labels$classes =  lapply( labels$tags , function(x){
  
   unlist(strsplit( as.character(x), ' ')) 
  
})

labels$numbers = unlist( lapply( labels$classes , function(x){
ret = ''
  for(i in 1:length(x)){
  ret = paste(ret, which(x[i] ==  c("haze",    "primary" ,      "agriculture" ,  "clear",   "water" ,    "habitation",    "road",    "cultivation" ,  "slash_burn",  "cloudy",    "partly_cloudy",     "conventional_mine", "bare_ground",  "artisinal_mine",   "blooming",  "selective_logging", "blow_down"))    )
}
  return(ret)
  
  }))

labels$classes = NULL

labels$temp = as.numeric(  unlist(lapply( labels$numbers , function(x){
sum( unlist( strsplit(x, ' ') ) ==2) > 0

  })  ))


write.csv(labels, file.path('db', 'labels.csv'))