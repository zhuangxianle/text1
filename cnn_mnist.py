# ##Head of CNN_TEST
#
#
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import os
# import matplotlib.pyplot as plt
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #解除警告
# tf.set_random_seed(1)
# np.random.seed(1)
#
# BATCH_SIZE = 64
# LR = 0.01              # learning rate
#
# # they has been normalized to range (0,1)
# mnist = input_data.read_data_sets('./mnist', one_hot=True)
# test_x = mnist.test.images[:2000]
# test_y = mnist.test.labels[:2000]
#
# # plot one example
# print(mnist.train.images.shape)     # (55000, 28 * 28)
# print(mnist.train.labels.shape)   # (55000, 10)
# plt.imshow(mnist.train.images[5].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[5])); plt.show()
#
# tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.
# image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)
# tf_y = tf.placeholder(tf.int32, [None, 10])            # input y
#
# # CNN
# # shape (28, 28, 1)
# conv1 = tf.layers.conv2d(inputs=image, filters=16, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu)
# # -> (28, 28, 16)
# pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
# # -> (14, 14, 16)
# conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
# # -> (14, 14, 32)
# pool2 = tf.layers.max_pooling2d(conv2, 2, 2)
# # -> (7, 7, 32)
# flat = tf.reshape(pool2, [-1, 7*7*32])
# # -> (7*7*32, )
# output = tf.layers.dense(flat, 10)              # output layer
#
# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
# train_op = tf.train.AdamOptimizer(LR).minimize(loss)
#
# # return (acc, update_op), and create 2 local variables
# accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
#
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # sess = tf.Session(config=config)
#
# sess = tf.Session()
# # the local var is for accuracy_op
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)     # initialize var in graph
#
# # following function (plot_with_labels) is for visualization, can be ignored if not interested
# # from matplotlib import cm
# # try: from sklearn.manifold import TSNE; HAS_SK = True
# # except: HAS_SK = False; print('\nPlease install sklearn for layer visualization\n')
# # def plot_with_labels(lowDWeights, labels):
# #     plt.cla(); X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
# #     for x, y, s in zip(X, Y, labels):
# #         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
# #     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)
# #
# # plt.ion()
# for step in range(3000):
#     b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
#     _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
#     if step % 50 == 0:
#         accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: test_x, tf_y: test_y})
#         print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.5f' % accuracy_)
#
#         # if HAS_SK:
#         #     # Visualization of trained flatten layer (T-SNE)
#         #     tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
#         #     low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
#         #     labels = np.argmax(test_y, axis=1)[:plot_only]; plot_with_labels(low_dim_embs, labels)
# # plt.ioff()
#
# # print 10 predictions from test data
# test_output = sess.run(output, {tf_x: test_x[:10]})
# pred_y = np.argmax(test_output, 1)
# print(pred_y, 'prediction number')
# print(np.argmax(test_y[:10], 1), 'real number')
#
#
# ##end of CNN_test

import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 100)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
#
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]