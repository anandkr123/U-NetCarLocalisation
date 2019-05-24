import tensorflow as tf
import json
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.io import imread, imshow
from skimage.transform import resize



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


# resizing the image, perform testing over it
new_img1 = resize(img_test, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
X_test[0] = new_img1
# print(X_test)

# storing images for training and their masks
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

# storing images for validation and their masks
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


def conv2d(input_tensor, depth, kernel, name, strides, padding="SAME"):
    return tf.layers.conv2d(input_tensor, filters=depth, kernel_size=kernel, strides=strides, padding=padding,
                            activation=tf.nn.relu, name=name)

def deconv2d(input_tensor, filter_size, output_size_r, output_size_c, out_channels, in_channels, name, strides):
    dyn_input_shape = tf.shape(input_tensor)
    batch_size = dyn_input_shape[0]
    out_shape = tf.stack([batch_size, output_size_r, output_size_c, out_channels])
    filter_shape = [filter_size, filter_size, out_channels, in_channels]
    w = tf.get_variable(name=name, shape=filter_shape)
    h1 = tf.nn.conv2d_transpose(input_tensor, w, out_shape, strides, padding='SAME')
    return h1


# model with  2 conv, 1 max-pool, 2 conv, 1 upconv, 2 conv, 1 deconv(final output)
net = conv2d(x, 32, 1, "Y0", strides=(1, 1)) # (198-2)/1  + 1 = 197  , (300-2)/1 + 1 = 299  (197,299)_

net = conv2d(net, 64, 3, "Y1", strides=(2, 2)) # (197-2)/1 + 1 = 196 , (196, 298)  (97, 149)

net = conv2d(net, 128, 3, "Y2", strides=(2, 2)) #32 (97-3)+1, 149-3 +1   (95, 147)

print(net.shape)


net = deconv2d(net, 1, 50, 75, 128, 128, "Y4_deconv", strides=[1, 1, 1, 1])
net = tf.nn.relu(net)

net = deconv2d(net, 2, 100, 150, 64, 128, "Y3_deconv", strides=[1, 2, 2, 1])
net = tf.nn.relu(net)

net = deconv2d(net, 2, 200, 300, 32, 64, "Y1_deconv", strides=[1, 2, 2, 1])
net = tf.nn.relu(net)

logits = deconv2d(net, 1, 200, 300, 1, 32, "logits_deconv", strides=[1, 1, 1, 1])

loss = tf.losses.sigmoid_cross_entropy(y_, logits)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_count = 1
display_count = 1

gr_loss = []
gr_epoch = []

for i in range(400):
    # training on batches of 10 images with 10 mask images
    if (batch_count > 5):
        batch_count = 1

    batch_X, batch_Y = next_batch(10, batch_count)

    batch_count += 1
    feed_dict = {x: batch_X, y_: batch_Y, lr: 0.0001}
    loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

    if (i % 5 == 0):
        print(str(display_count) + " training loss:", str(loss_value))
        display_count += 1
# perform validation after every 5 complete set of training
    if i % 19 == 0:
        val_batch_X, val_batch_Y = val_images[:73], val_labels[0:73]
        feed_dict = {x: val_batch_X, y_: val_batch_Y}
        val_loss = sess.run(loss, feed_dict=feed_dict)
        print('Validation set loss is', val_loss)

print("Done!")

test_image = X_test[0]

test_image = np.reshape(test_image, [-1, 200, 300, 1])
test_data = {x: test_image}
test_mask = sess.run([logits], feed_dict=test_data)
test_mask = np.reshape(np.squeeze(test_mask), [IMG_HEIGHT, IMG_WIDTH, 1])


for i in range(IMG_HEIGHT):
    for j in range(IMG_WIDTH):
            test_mask[i][j] = int(sigmoid(test_mask[i][j])*255)

fig = plt.figure()
plt.subplot(1, 3, 1)
imshow(img_test)
plt.subplot(1, 3, 2)
imshow(test_mask.squeeze().astype(np.uint8))
plt.subplot(1, 3, 3)
imshow(img_test_mask)
plt.show()
