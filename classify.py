#%%

# coding: utf-8

# In[145]:


from __future__ import absolute_import, division, print_function


# In[146]:


import numpy as np
import tensorflow as tf
from keras.datasets import cifar10



import matplotlib
import matplotlib.pyplot as plt



(features, labels), (X_test_orig, Y_test_orig) = cifar10.load_data()
features = features/255 - 0.5
labels = labels.flatten()

test_images = X_test_orig/255 - 0.5
test_labels = Y_test_orig.flatten()


labels = labels.flatten()



# In[152]:


height = 32
width = 32
channels = 3
n_inputs = height * width


# In[153]:


tf.reset_default_graph()


# In[154]:


def initialize_parameters():
    height = 32
    width = 32
    channels = 3
    n_inputs = height * width
    X = tf.placeholder(tf.float32, shape=[None,  height, width, channels], name="X")
    
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    return X, y


# In[155]:


def get_next_batch(features, labels, train_size, batch_index, batch_size):
    training_images = features[:train_size,:,:]
    training_labels = labels[:train_size]
    
    start_index = batch_index * batch_size
    end_index = start_index + batch_size

    return features[start_index:end_index,:,:], labels[start_index:end_index]


# In[156]:


def forward_propagation(X, training=True):
  
    # Convolutional Layer #1
    dropout_rate = 0.3

    training = tf.placeholder_with_default(False, shape=(), name='training')
    X_drop = tf.layers.dropout(X, dropout_rate, training=training)
    print(X_drop)
    conv1 = tf.layers.conv2d(
      inputs=X_drop,
      filters=32,
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(conv1, filters=64, 
                       kernel_size=3,
                       strides=2, padding="valid",
                       activation=tf.nn.relu, name="conv2")
    pool3 = tf.nn.max_pool(conv2,
                     ksize=[1, 2, 2, 1],
                     strides=[1, 2, 2, 1],
                     padding="VALID")
    conv4 = tf.layers.conv2d(pool3, filters=128, 
                       kernel_size=4,
                       strides=3, padding="SAME",
                       activation=tf.nn.relu, name="conv4")

    pool5 = tf.nn.max_pool(conv4,
                     ksize=[1, 2, 2, 1],
                     strides=[1, 1, 1, 1],
                     padding="VALID")

    pool5_flat = tf.contrib.layers.flatten(pool5)

    fullyconn1 = tf.layers.dense(pool5_flat, 128,
                           activation=tf.nn.relu, name="fc1")

    fullyconn2 = tf.layers.dense(fullyconn1, 64,
                           activation=tf.nn.relu, name="fc2")
    logits = tf.layers.dense(fullyconn2, 10, name="output")
    return logits, training


# In[157]:





# In[158]:


tf.reset_default_graph()

n_epochs = 1
batch_size = 128
X, y = initialize_parameters()
logits, training = forward_propagation(X)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=y)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)) 


# In[159]:


def model(features, labels, test_images, test_labels):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
    #         batch_index = 0
            # Add this in when we want to run the training on all batches in CIFAR-10
            batch_index = 0

            train_size = int(len(features))

            for iteration in range(train_size // batch_size):
                X_batch, y_batch = get_next_batch(features, 
                                                                            labels, 
                                                                            train_size, 
                                                                            batch_index,
                                                                            batch_size)
                batch_index += 1

                sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})

            acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
            acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels})
            print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
            print ("Train Accuracy:", round(acc_train, 2)*100,"%",  end="\t")
            print ("Test Accuracy:", round(acc_test, 2)*100,"%", end="\n")
        import os
        if not os.path.exists("ADL"):
            os.makedirs("ADL")
        model_saver = tf.train.Saver()
        model_saver.save(sess, "ADL/cifar_model")


# In[160]:

#%%
#parameters = model(X_train, Y_train, X_test, Y_test)
#%%
import cv2
import sys
if __name__ == "__main__":
    if sys.argv[1] == 'test' and len(sys.argv) < 3: 
        print("Missing argument for image file")
    elif sys.argv[1] == "test":
        #tf.reset_default_graph()
        with tf.Session() as session:
            import tensorflow as tf

            #tf.reset_default_graph()
            with tf.Session() as sess:
                # Restore variables from disk.
                saver = tf.train.import_meta_graph('cifar_model.meta')
                saver.restore(sess,"cifar_model")
                
                print("Model restored.")
                acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels})
                
                predicition = sess.run(p, feed_dict={x: features[0:10]})
                print(predicition)
                img = cv2.imread(sys.argv[2])
                img = cv2.resize(img, (32, 32))
                resized_image_flatten = img.reshape(img.shape[0]*img.shape[1]*img.shape[2], 1)
                classnames = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
                print("Your algorithm predicts: y = " + classnames[np.squeeze(predict(resized_image_flatten, parameters))])
    elif sys.argv[1] == "train":
         model(features, labels, test_images, test_labels)
