source('packages.r')

labels = readRDS('db/labels.rds')

im = readImage(  paste0('db/train-jpg/',labels$image_name[i], '.jpg' ) )