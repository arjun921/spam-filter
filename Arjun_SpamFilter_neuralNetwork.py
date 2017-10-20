import tensorflow as tf
import pickle
import numpy as np


with open('data/data.pickle', 'rb') as f:
        tr_data, tr_label, tst_data, tst_label = pickle.load(f)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
n_batches = 10

x = tf.placeholder('float', [None, len(tr_data[0])])
y = tf.placeholder('float')

# Neural Network Model
def neural_network(data):
    hd_layer1 = {'weights': tf.Variable(tf.random_normal([len(tr_data[0]), n_nodes_hl1])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hd_layer2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hd_layer3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output    = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                 'biases': tf.Variable(tf.random_normal([n_classes]))}

    # (input_data*weight) + biases
    l1  = tf.add(tf.matmul(data,hd_layer1['weights']),hd_layer1['biases'])
    l1 = tf.nn.relu(l1) # activation function - threshold function

    l2  = tf.add(tf.matmul(l1,hd_layer2['weights']),hd_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3  = tf.add(tf.matmul(l2,hd_layer3['weights']),hd_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output  = tf.matmul(l3,output['weights']) + output['biases']

    return output

# Code to train the neural netowrk
def train_neural_network(x):
    # gives predicted value based on input data
    prediction = neural_network(x)
    # gives cost function (difference) between predicted and actual output
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    # Stochastic gradient descent
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    # no. of cycles of feed forward + backprop
    hm_epochs= 10

    # Run tensorFlow
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in range(0,len(tr_data),n_batches):
                x_batch = np.array(tr_data[i:i+100])
                y_batch = np.array(tr_label[i:i+100])
                _, c= sess.run([optimizer,cost],feed_dict={x:x_batch,y:y_batch})
                epoch_loss += c
            print('Epoch: {} completed out of:  {} loss: {}'.format(epoch,hm_epochs,epoch_loss))

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print(accuracy.eval({x:tst_data,y:tst_label}))

train_neural_network(x)
