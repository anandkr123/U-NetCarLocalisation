import tensorflow as tf
import json
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize


# image parameters

IMG_WIDTH = 300
IMG_HEIGHT = 200
IMG_CHANNELS = 1
TRAIN_IMAGES = 50
TEST_IMAGES =73
ext = ".png";


# prepare a circular mask at the location
def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    masked = dist_from_center <= radius
    return masked

# method for reading a json file from the directory
def getJSON(filePathAndName):
    with open(filePathAndName, 'r') as fp:
        return json.load(fp)

# get the next batch of images
def next_batch(batch_s, iters):
    count = batch_s * iters
    return images[count-batch_s:(count-1)], labels[count-batch_s:(count-1)]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# contains all train images and masks
images = np.zeros((TRAIN_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros((TRAIN_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.bool)

# contains all validation images and masks
val_images = np.zeros((TEST_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
val_labels = np.zeros((TEST_IMAGES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.bool)

# image to test
X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

# JSON file to read path of train an test images
myObj_train = getJSON('../caltech101_car_side/training.json')
myObj_test = getJSON('../caltech101_car_side/test.json')

cnt = 1
count = 1


# specify the test image
test_path = '/home/smaph/PycharmProjects/CarLocalisation/caltech101_car_side/all/image_0007.jpg'
img_test = imread(test_path)

# specify the center to create the test-mask
center = np.array([144, 116])
mask = create_circular_mask(200, 300, center, 16)
new_mask = mask*255
cv2.imwrite('/home/smaph/PycharmProjects/CarLocalisation/caltech101_car_side/all/new.png', new_mask)
test_mask_path = '/home/smaph/PycharmProjects/CarLocalisation/caltech101_car_side/all/new.png'
img_test_mask = imread(test_mask_path)


# resizing the image, to perform testing over it
new_img1 = resize(img_test, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
X_test[0] = new_img1
# print(X_test)

# storing images for training and their masks from the JSON file itself
for data in myObj_train:
    rel_path = data.get("image_path")
    home_path = '/home/smaph/PycharmProjects/CarLocalisation/caltech101_car_side/'
    mas_path = "train_mask/" + str(cnt) + ext
    img = imread(home_path + rel_path)
    mas = imread(home_path + mas_path)
    xtemp = data.get("object_instances")[0].get("reference_point")[0]
    ytemp = data.get("object_instances")[0].get("reference_point")[1]
# resizing the image
    new_img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    new_mas = resize(mas, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    images[cnt-1] = new_img
    labels[cnt-1] = new_mas
    cnt += 1


cnt = 1

# storing images for validation reading from JSON file and their masks
for data in myObj_test:
    rel_path = data.get("image_path")
    home_path = '/home/smaph/PycharmProjects/CarLocalisation/caltech101_car_side/'
    mas_path = "test_mask/" + str(cnt) + ext
    img = imread(home_path + rel_path)
    mas = imread(home_path + mas_path)
    xtemp = data.get("object_instances")[0].get("reference_point")[0]
    ytemp = data.get("object_instances")[0].get("reference_point")[1]
    # resizing the image
    new_img = resize(img, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    new_mas = resize(mas, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    val_images[cnt-1] = new_img
    val_labels[cnt-1] = new_mas
    cnt += 1




x = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
y_ = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
lr = tf.placeholder(tf.float32)

# conv with batch normalization

def conv2d(input_tensor, depth, kernel, name, strides, padding="SAME"):
    conv = tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding,
                             activation=tf.nn.relu, name=name)
    return tf.layers.BatchNormalization()(conv)

# deconvolution
def deconv2d(input_tensor, filter_size, output_size_r, output_size_c, out_channels, in_channels, name, strides):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size_r, output_size_c, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h2 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    h1 = tf.nn.relu(h2)
    return h1

# U- NET ARCHITECTURE with two times contraction

cnet1 = conv2d(x,16,3,"conv1",strides=(1,1))  #(198,298)
cnet2 = conv2d(cnet1,16,3,"conv2",strides=(1,1))

                                                                                # first contraction (MAX-POOL)
cmax1 = tf.nn.max_pool(cnet2, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

cnet3 = conv2d(cmax1,32,3,"conv3",strides=(1,1))
cnet4 = conv2d(cnet3,32,3,"conv4",strides=(1,1))

                                                                                # second contraction (MAX-POOL)
cmax2 = tf.nn.max_pool(cnet4, ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

cnet5 = conv2d(cmax2,64,3,"conv5",strides=(1,1))
cnet6 = conv2d(cnet5,64,3,"conv6",strides=(1,1))

                                                                                # first up-sampling
dnet1 = deconv2d(cnet6, 2, 100, 150, 32, 64, "dconv1", strides=[1, 2, 2, 1])
merge11 = tf.concat(values=[cnet4, dnet1], axis=-1)                             # concat the feature maps with the last contraction
# print(merge11.shape)

cnet7=conv2d(merge11,32,3,"conv7",strides=[1,1])
cnet8=conv2d(cnet7,32,3,"conv8",strides=[1,1])
                                                                                # second up-sampling
dnet2 = deconv2d(cnet8, 2, 200, 300, 16, 32, "dconv2", strides=[1, 2, 2, 1])
merge12 = tf.concat(values=[cnet2, dnet2], axis=-1)                             # concat the feature maps with the second last contraction
# print(merge12.shape)

cnet9=conv2d(merge12,32,3,"conv9",strides=[1,1])
cnet10=conv2d(cnet9,16,3,"conv10",strides=[1,1])

                                                                                 # logits with sigmoid(not reLu) to apply binary-crossentropy
logits=tf.layers.conv2d(cnet10, filters=1, kernel_size=1, strides=[1,1], padding='SAME',
                             activation=tf.nn.sigmoid, name='logits')

loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_, logits))           # performing loss over the complete image
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Saving only recent last 2 checkpoints
saver = tf.train.Saver(max_to_keep=2)

batch_count = 1
display_count = 0

gr_val = []
gr_epoch = []
min_delta= 0.001

# returns a CheckpointState if the state was available, None otherwise
ckpt = tf.train.get_checkpoint_state('./')


if ckpt:
    print('Reading last checkpoint....')
    saver = tf.train.import_meta_graph('u-net-model-50.meta')

# restoring the model from the last checkpoint
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print('Model restored')
else:
    print('Creating the checkpoint and models')

for i in range(400):
    # training on batches of 10 images with 10 mask images
    if (batch_count > 5):
        batch_count = 1

    batch_X, batch_Y = next_batch(10, batch_count)

    batch_count += 1
    feed_dict = {x: batch_X, y_: batch_Y, lr: 0.0001}
    loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

    if i % 4 == 0:
        display_count += 1
        print('After Epoch :', str(display_count) + " training loss:", str(loss_value))
        gr_epoch.append(loss_value)
# perform validation after every epoch of training
    if i % 4 == 0:
        val_batch_X, val_batch_Y = val_images[:20], val_labels[0:20]
        feed_dict = {x: val_batch_X, y_: val_batch_Y}
        val_loss = sess.run(loss, feed_dict=feed_dict)
        gr_val.append(val_loss)
        print('After Epoch {}, Loss on validation set {}'.format(display_count, val_loss))
# saving the model after every 50 iterations
    if i % 50 == 0:
        saver.save(sess, './u-net-model', global_step=i, write_meta_graph=False)

        # EARLY STOPPING
    # if display_count > 1:
    #     if abs(gr_val[display_count-2]-gr_val[display_count-1]) < min_delta:
    #         print(gr_val[display_count-2])
    #         print(gr_val[display_count-1])
    #         break


# print("Validation loss didn't showed improvement less than 1% ")

test_image = X_test[0]
# Taking a test image to localize the area
test_image = np.reshape(test_image, [-1, 200, 300, 1])
test_data = {x: test_image}
# getting the logits
test_mask = sess.run([logits], feed_dict=test_data)
# removing the channel dimension(to display the result)
test_mask = np.reshape(np.squeeze(test_mask), [IMG_HEIGHT, IMG_WIDTH, 1])

# converting the image pixels to int(0-255) range values
for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
            test_mask[i][j] = int((test_mask[i][j])*255)

fig = plt.figure()
                     # The original image
plt.subplot(1, 3, 1)
imshow(img_test)
                     # The localised image( with RoI localisation )
plt.subplot(1, 3, 2)
imshow(test_mask.squeeze().astype(np.uint8))
                     # The masked image (with localised circle )
plt.subplot(1, 3, 3)
imshow(img_test_mask)
plt.show()
