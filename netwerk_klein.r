source('packages.r')
source('read_batch.r')

path = "/home/daniel/R/Amazon_chalange/db"

#read in file with labels and file names
data = readRDS( file.path( path, 'labels.rds') )

#split in train and test
split = sample(x =  c(1:nrow(data)), size = round(0.8*nrow(data)) )
train = data[split,]
test = data[-split,]

train_class_1 = train[train$temp == 0,]
train_class_2 = train[train$temp == 1,]

test_class_1 = train[train$temp == 0,]
test_class_2 = train[train$temp == 1,]

#set parameters
aantal_pool  = 2
aantal_kanalen = 64
clas = as.integer(2)#number of classes
schaal = 1 #scale of pixel values

out = 0.5 #dropout
batch_train = 2 #batchsize
batch_test = 50 #batchsize test
ds = 0.99999 #gradient descent
lr = 1e-4 #learningrate

h = as.integer(256) #heigth image
w = as.integer(256) #width image
kanalen = as.integer(3) #chanals of image

#Place holders
x <- tf$placeholder(tf$float32, shape(NULL, h,w,kanalen), 'x')
#target values
labels <- tf$placeholder(tf$int64, shape(NULL), 'labels')
#dropout rate
keep_prob <- tf$placeholder(tf$float32, shape(),'keep_prob')
#learningrate
lrate <- tf$placeholder(tf$float32, shape(), 'lrate')




#Define variables
w_conv1 <-  tf$Variable( tf$truncated_normal(shape(4L, 4L, 3L, 16L),stddev=0.1), 'w_conv1')
b_conv1 <- tf$Variable( tf$truncated_normal( shape(16L),stddev=0.1), 'b_conv1')

w_conv2 <- tf$Variable( tf$truncated_normal(shape = shape(4L, 4L, 16L, 32L), stddev=0.1), 'w_conv2')
b_conv2 <- tf$Variable( tf$truncated_normal(shape = shape(32L), stddev=0.1), 'b_conv2')

w_conv3 <- tf$Variable( tf$truncated_normal(shape = shape(4L, 4L, 32L, 64L), stddev=0.1), 'w_conv3')
b_conv3 <- tf$Variable( tf$truncated_normal(shape = shape(64L), stddev=0.1), 'b_conv3')

w_fc1 <- tf$Variable( tf$truncated_normal(shape((w*h)/(4^(aantal_pool)) * aantal_kanalen, 1024L), stddev=0.1), 'w_fc1')
b_fc1 <- tf$Variable( tf$truncated_normal( shape(1024L) , stddev=0.1), 'b_fc1')

w_fcout <- tf$Variable( tf$truncated_normal( shape(1024L, clas)  , stddev=0.1), 'w_fcout')
b_fcout <- tf$Variable( tf$truncated_normal( shape(clas) , stddev=0.1), 'b_fcout')


#Define network
h_conv1 <- tf$nn$relu( tf$nn$conv2d(x, w_conv1 , strides=c(1L, 1L, 1L, 1L), padding='SAME') + b_conv1)
h_pool1 <- tf$nn$max_pool(h_conv1, ksize=c(1L, 2L, 2L, 1L),strides=c(1L, 2L, 2L, 1L), padding='SAME')
h_conv2 <- tf$nn$relu( tf$nn$conv2d(h_pool1, w_conv2, strides=c(1L, 1L, 1L, 1L), padding='SAME') + b_conv2)
h_pool2 <- tf$nn$max_pool(h_conv2, ksize=c(1L, 2L, 2L, 1L),strides=c(1L, 2L, 2L, 1L), padding='SAME')
h_conv3 <- tf$nn$relu( tf$nn$conv2d(h_pool2, w_conv3, strides=c(1L, 1L, 1L, 1L), padding='SAME') + b_conv3)
h_conv3_flat <- tf$reshape( h_conv3 , shape(-1L, (w*h)/(4^(aantal_pool)) * aantal_kanalen))
h_fc1 <- tf$nn$relu(tf$matmul(h_conv3_flat, w_fc1) + b_fc1)
h_fcout <- tf$matmul(h_fc1, w_fcout) + b_fcout
h_output <- tf$nn$softmax(h_fcout)



#Define cost function
cost = tf$reduce_mean( tf$nn$softmax_cross_entropy_with_logits( logits = h_fcout, labels = tf$one_hot(labels, clas)) )

#Define optimizer
train_step <- tf$train$AdamOptimizer(lrate)$minimize(cost)

#Define some variables to print along the way
correct_prediction <- tf$equal(tf$argmax(h_output, 1L),labels)
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))

#Define the session
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())

#Train the network
for (i in 1:200000) {
  
  #lees 50 random plaatjes in
  samp = sample( x=  c(1: nrow(train_class_1)) , size = batch_train )
  batch_class_1 = train_class_1[samp,]
  samp = sample( x=  c(1: nrow(train_class_2)) , size = batch_train)
  batch_class_2 = train_class_2[samp,]
  batch = rbind(batch_class_1, batch_class_2)
  batch_labels = as.vector(batch$temp)
  batch_files= read_batch(files = batch$image_name)
  
  
  #train met gradient descent
  sess$run(train_step, feed_dict = dict(x = batch_files , labels = batch_labels , keep_prob = out, lrate = ds^i*lr))
  
  #  sess$run(cost, feed_dict = dict(x = batch_files , labels = batch_labels , keep_prob = out, lrate = ds^i*lr))
  
  
  
  #valideer om de 100 keer hoe het gaat op de testset
  if (i %% 300 == 0) {
    samp = sample( x=  c(1: nrow(train_class_1)) , size = batch_test )
    batch_class_1 = train_class_1[samp,]
    samp = sample( x=  c(1: nrow(train_class_2)) , size = batch_test)
    batch_class_2 = train_class_2[samp,]
    batch = rbind(batch_class_1, batch_class_2)
    batch_labels = as.vector(batch$temp)
    batch_files= read_batch(files = batch$image_name)
    
    
    train_accuracy =  sess$run(accuracy, feed_dict = dict(x = batch_files , labels = batch_labels , keep_prob = 1))
    print( paste("step:", i, "train accuracy:", train_accuracy) ) 
    
    #evalueer op de testset
    samp = sample( x=  c(1: nrow(test_class_1)) , size = batch_test )
    batch_class_1 = test_class_1[samp,]
    samp = sample( x=  c(1: nrow(test_class_2)) , size = batch_test)
    batch_class_2 = test_class_2[samp,]
    batch = rbind(batch_class_1, batch_class_2)
    batch_labels = as.vector(batch$temp)
    batch_files= read_batch(files = batch$image_name)
    
  
    test_accuracy =  sess$run(accuracy, feed_dict = dict(x = batch_files , labels = batch_labels , keep_prob = 1))


    print( paste("step:", i, "test accuracy:", test_accuracy) )
  }
  
  
  
  
  
  
  
}

