import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageOps
from scipy.ndimage.filters import gaussian_filter
import random

#number of classes
nClass=32

#simple model (set to True) or convolutional neural network (should set to False)
simpleModel=False

#dimensions of image (pixels)
height= 64
width= 64

#data6 is HOMUS dataset, data4 is mixture dataset(sp)
tfrecords_train_filename = 'data6/train-00000-of-00001'
tfrecords_test_filename  = 'data6/validation-00000-of-00001'
tfrecords_test_sp_filename  = 'data4/validation-00000-of-00001'

#train batch size
trainBatchSize = 128
#test batch size
testBatchSize = 3090
#test_sp batch size
test_sp_BatchSize = 2718

#very small offset for cross-entropy loss function (avoid overfitting problem)
episo_offset = 1e-8

#save paths
ckpt_path = './model/test-model.ckpt'

# Function to tell TensorFlow how to read a single image from input file
def getImage(filename, req_batchSize):
    # convert filenames to a queue for an input pipeline.
 
    # object to read records
    recordReader = tf.TFRecordReader()

    # read the full set of features for a single example 
    key, fullExample = recordReader.read(filename)

    # parse the full example into its' component features.
    features = tf.parse_single_example(
        fullExample,
        features={
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/channels':  tf.FixedLenFeature([], tf.int64),            
            'image/class/label': tf.FixedLenFeature([],tf.int64),
            'image/class/text': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/format': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/filename': tf.FixedLenFeature([], dtype=tf.string,default_value=''),
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')
        })


    # now we are going to manipulate the label and image features
    label = features['image/class/label']
    image_buffer = features['image/encoded']

    # Decode the jpeg
    with tf.name_scope('decode_jpeg',[image_buffer], None):
        # decode
        image = tf.image.decode_jpeg(image_buffer, channels=3)
    
        # and convert to single precision data type
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)


    # cast image into a single array, where each element corresponds to the greyscale
    # value of a single pixel. 
    # the "1-.." part inverts the image, so that the background is black.

    image=tf.reshape(1-tf.image.rgb_to_grayscale(image),[height*width])

    # re-define label as a "one-hot" vector 
    # it will be [0,1] or [1,0] here. 
    # This approach can easily be extended to more classes.
    label=tf.stack(tf.one_hot(label-1, nClass))


    images, labels = tf.train.shuffle_batch( [image, label],
                                             batch_size= req_batchSize,
                                             capacity=8192,
                                             num_threads=2,
                                             min_after_dequeue=600)

    return images, labels

def plot_image(image):
    plt.imshow(image.reshape(height,width), cmap = 'binary')
    plt.show()


# images shape(batch_sizes, image1D_array)
def augmentation(images):
    _batch_size = (images.shape[0])
    for i in range(_batch_size):
        tmp_arr = images[i].reshape(height,width)
        im = Image.fromarray(tmp_arr)

        #offset
        ratio = random.uniform(1,2) #64/1~64/2
        scaled_w = round(width//ratio)
        scaled_h = round(height//ratio)
        if (scaled_w % 2 == 1):
            scaled_w = scaled_w + 1
            scaled_h = scaled_h + 1
        nim = im.resize( (scaled_w, scaled_h), Image.LANCZOS )
        #example: im2 =ImageOps.expand(nim, (16,16,16,16), fill=0) #distance to (L,T,R,B)
        residue_edge_h = (height - abs(scaled_h)) //2
        residue_edge_w = (width - abs(scaled_w))  //2
        offset_range = min(residue_edge_h,residue_edge_w)
        offset_h = random.randint(-offset_range,offset_range)
        offset_w = random.randint(-offset_range,offset_range)
        im2 =ImageOps.expand(nim, (residue_edge_w+offset_w,residue_edge_h+offset_h,residue_edge_w-offset_w,residue_edge_h-offset_h), fill=0)

        #random rotate
        degree_angle = random.randint(-15,15) # In degrees
        im2 = im2.rotate(degree_angle) #expand = false

        np_im2 = np.array(im2) #back to np-array
        
        np_im2 = np.array(im2)
        if(i%3==0): #do blur
            sigma_val = random.uniform(0.2,1)
            np_im2 = gaussian_filter(np_im2, sigma=sigma_val)
        if(i%3==1): #additional guassian noise points
            mean = 0
            #var = 0.01~0.001
            var = random.uniform(0.001,0.005)
            sigma_val = var**0.5
            gauss = np.random.normal(mean,sigma_val,(height,width))
            gauss = gauss.reshape(height,width)
            np_im2 = np_im2 + gauss
        
        mod_image = np_im2.reshape(height*width)
        mod_image[mod_image > 1] = 1
        mod_image[mod_image < 0] = 0
        images[i] = mod_image
        #mod_image.imshow(mod_image, cmap = 'binary')
        #mod_image.show()
    return images


# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# x is the input array, which will contain the data from an image 
# this creates a placeholder for x, to be populated later
x = tf.placeholder(tf.float32, [None, width*height])
# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, nClass])

if simpleModel:
    # run simple model y=Wx+b given in TensorFlow "MNIST" tutorial
    print("Running Simple Model y=Wx+b")

    # initialise weights and biases to zero
    # W maps input to output so is of size: (number of pixels) * (Number of Classes)
    W = tf.Variable(tf.zeros([width*height, nClass]))
    # b is vector which has a size corresponding to number of classes
    b = tf.Variable(tf.zeros([nClass]))

    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    y = tf.nn.softmax(tf.matmul(x, W) + b)

else:
    # run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial

    # functions to init small positive weights and biases
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

  # set up "vanilla" versions of convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    print ("Running Convolutional Neural Network Model")
    nFeatures1 = 32
    nFeatures2 = 140
    nNeuronsfc = 1024
    conv1_filter_size = 3
    conv2_filter_size = 3

    # use functions to init weights and biases
    # nFeatures1 features for each patch of size 5x5
    # SAME weights used for all patches
    # 1 input channel
    W_conv1 = weight_variable([conv1_filter_size, conv1_filter_size, 1, nFeatures1])
    b_conv1 = bias_variable([nFeatures1])
  
    # reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth 
    # means 1 colour channel per pixel
    # x_image = tf.reshape(x, [-1,28,28,1])
    x_image = tf.reshape(x, [-1,width,height,1])
  
    # hidden layer 1 
    # pool(convolution(Wx)+b)
    # pool reduces each dim by factor of 2.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # similarly for second layer, with nFeatures2 features per 5x5 patch
    # input is nFeatures1 (number of features output from previous layer)
    W_conv2 = weight_variable([conv2_filter_size, conv2_filter_size, nFeatures1, nFeatures2])
    b_conv2 = bias_variable([nFeatures2])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
  
    # denseley connected layer. Similar to above, but operating
    # on entire image (rather than patch) which has been reduced by a factor of 4 
    # in each dimension
    # so use large number of neurons 

    # check our dimensions are a multiple of 4
    #if (width%4 or height%4):
    #  print "Error: width and height must be a multiple of 4"
    #  sys.exit(1)
      
    W_fc1 = weight_variable([(width//4) * (height//4) * nFeatures2, nNeuronsfc])
    b_fc1 = bias_variable([nNeuronsfc])
      
    # flatten output from previous layer
    h_pool2_flat = tf.reshape(h_pool2, [-1, (width//4) * (height//4) * nFeatures2])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
      
    # reduce overfitting by applying dropout
    # each neuron is kept with probability keep_prob
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
      
    # create readout layer which outputs to nClass categories
    W_fc2 = weight_variable([nNeuronsfc, nClass])
    b_fc2 = bias_variable([nClass])
      
    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    # this is not run until later
    y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# measure of error of our model for both models
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y+episo_offset), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#input batch take from resources
train_filenameQ = tf.train.string_input_producer([tfrecords_train_filename],num_epochs=None)
imageBatch, labelBatch = getImage(train_filenameQ, trainBatchSize) 

test_filenameQ = tf.train.string_input_producer([tfrecords_test_filename],num_epochs=None)
vimageBatch, vlabelBatch = getImage(test_filenameQ, testBatchSize)

test_sp_filenameQ = tf.train.string_input_producer([tfrecords_test_sp_filename],num_epochs=None)
vimageBatch_sp, vlabelBatch_sp = getImage(test_sp_filenameQ, test_sp_BatchSize) 

# initialize the variables
sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
#for early stop
last_valid_rate = 0.0 
last_valid_sp_rate = 0.0
max_valid_rate = 0.0
max_valid_sp_rate = 0.0 

# start the threads used for reading files
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

iter_list = list()
loss_list = list()
loss_sp_list = list()
loss_v_list = list()
acc_list = list()
acc_sp_list = list()
acc_v_list = list()

# start training
# try to read batch  and try to see content
nSteps=15000
for i in range(nSteps):
    batch_xs, batch_ys = sess.run([imageBatch, labelBatch])
    #augmentation
    mod_batch_xs = augmentation(batch_xs)
    
    if simpleModel:
        train_step.run(feed_dict={x: mod_batch_xs, y_: batch_ys})
    else:
        train_step.run(feed_dict={x: mod_batch_xs, y_: batch_ys, keep_prob: 0.5})
    
    if i==0 or ((i+1)%100 == 0): # 200steps, then perform validation 
    # get a validation batch
        vbatch_xs, vbatch_ys = sess.run([vimageBatch, vlabelBatch])
        vbatch_sp_xs,vbatch_sp_ys = sess.run([vimageBatch_sp, vlabelBatch_sp])
        if simpleModel:
            valid_accuracy = accuracy.eval(feed_dict={x:vbatch_xs, y_: vbatch_ys})
            train_accuracy = accuracy.eval(feed_dict={x:mod_batch_xs, y_: batch_ys})
        else:
            valid_accuracy = accuracy.eval(feed_dict={x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
            valid_accuracy_sp = accuracy.eval(feed_dict={x:vbatch_sp_xs, y_: vbatch_sp_ys, keep_prob: 1.0})
            train_accuracy = accuracy.eval(feed_dict={x:mod_batch_xs, y_: batch_ys, keep_prob: 1.0})

            loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x:mod_batch_xs, y_: batch_ys, keep_prob: 1.0})
            loss_sp, acc_sp = sess.run([cross_entropy, accuracy], feed_dict={x:vbatch_sp_xs, y_: vbatch_sp_ys, keep_prob: 1.0})
            loss_v, acc_v = sess.run([cross_entropy, accuracy], feed_dict={x:vbatch_xs, y_: vbatch_ys, keep_prob: 1.0})
            iter_list.append(i+1)
            loss_list.append(loss)
            loss_sp_list.append(loss_sp)
            loss_v_list.append(loss_v)
            acc_list.append(acc)
            acc_sp_list.append(acc_sp)
            acc_v_list.append(acc_v)
            
        print("step %d, validation accuracy %g"%(i+1, valid_accuracy))
        print("step %d, validation_sp accuracy %g"%(i+1, valid_accuracy_sp))
        print("step %d, training accuracy %g"%(i+1, train_accuracy))

        if valid_accuracy > 0.80 and valid_accuracy_sp > 0.68 and valid_accuracy_sp > max_valid_sp_rate:
            saver.save(sess, ckpt_path , global_step=1) #record training model
		
        if valid_accuracy - last_valid_rate < -0.5 or valid_accuracy_sp - last_valid_sp_rate < -0.5 :
            print("early stop")
            break  #early stop
        if valid_accuracy > max_valid_rate:
            max_valid_rate = valid_accuracy

        if valid_accuracy_sp > max_valid_sp_rate:
            max_valid_sp_rate = valid_accuracy_sp

        last_valid_rate = valid_accuracy
        last_valid_sp_rate = valid_accuracy_sp


# finalise 
coord.request_stop()
coord.join(threads)

print("max valid_sp rate: %g" %(max_valid_sp_rate))
print("max valid rate: %g" %(max_valid_rate))
print("nFeatures1: %d" %(nFeatures1))
print("nFeatures2: %d" %(nFeatures2))
print("nNeuronsfc : %d" %(nNeuronsfc))
print("conv1_filter_size : %d" %(conv1_filter_size))
print("conv2_filter_size : %d" %(conv2_filter_size))

#output list value
'''
print(iter_list)
print(loss_list)
print(acc_list)
print(loss_sp_list)
print(acc_sp_list)
print(loss_v_list)
print(acc_v_list)
'''

#close the session to release resources
sess.close()



