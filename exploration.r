source('packages.r')

files = list.files('db/train-jpg')

labels = read.csv('db/train_v2.csv')
labels$class = as.numeric(labels$tags)


for(i in 1:10){
  
  im = readImage( file.path('db', 'train-jpg', paste0(labels$image_name[i], '.jpg') ))
  plot(im)
  print(labels$tags[i])
}