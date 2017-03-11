import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load traffic signs data.
training_file = "./train.p"
with open(training_file, mode='rb') as f:
    train = pickle.load(f)

X = train['features']
Y = train['labels']
print("X.shape: ", X.shape)
print("Y.shape: ", Y.shape)

sign_names = pd.read_csv('signnames.csv')
print("sign_names: ", sign_names)

nb_classes = len(np.unique(Y))
print("Unique labels count: ", nb_classes)

# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.3, random_state=42)
print("X_train.shape: ", X_train.shape)
print("y_train.shape: ", y_train.shape)
print("X_valid.shape: ", X_valid.shape)
print("y_valid.shape: ", y_valid.shape)

nb_examples = len(y_train)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, nb_classes)

resized_img = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized_img, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=0.1))
fc8b = tf.Variable(tf.zeros(nb_classes))

logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

learning_rate = 0.0005
epochs = 10
batch_size = 128

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss, var_list=[fc8W, fc8b])

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# TODO: Train and evaluate the feature extraction model.
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Training
print("Training...")
print()

for i in range(epochs):
    
    # Shuffle trained data
    X_train, y_train = shuffle(X_train, y_train)
    
    for offset in range(0, nb_examples, batch_size):
        end = offset + batch_size
        batch_x, batch_y = X_train[offset:end], y_train[offset:end]
        sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

    # Evaluation
    validation_accuracy = evaluate(X_valid, y_valid, sess)
    print("EPOCH {} ...".format(i+1))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print()

sess.close()
