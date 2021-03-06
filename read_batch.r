read_batch = function(files){
  
  
  batch = array(0, dim = c(length(files), w, h, 3))
  for(i  in 1:length(files)){
    file = files[i]
    im = readJPEG( file.path(path, paste0( 'train-jpg/' ,file, '.jpg')))
    batch[i,,,] = im[,,1:3]
  }
  return(batch)
}