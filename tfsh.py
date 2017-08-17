# TensorFlow Shorthands: A convinience high level API to tensorflow
# Warning: This API has NOT yet been stabelized and may vary from project to project

import numpy as np
import tensorflow as tf

#%%
def get_weight(name, shape, dtype=tf.float32, mean=0.0, stddev=0.1):
    return tf.get_variable(name, shape, dtype, tf.truncated_normal_initializer(mean,stddev))

def get_bias(name, shape, dtype=tf.float32, value=0.1):
    return tf.get_variable(name, shape, dtype, tf.constant_initializer(value))

def get_action_mask(i, n):
    return np.eye(n)[i]


#%%
def sess_saver():
    saver_params = {}
    for i in tf.trainable_variables():
        i_name = i.name[0:i.name.index(':')]
        saver_params[i_name] = i
    return tf.train.Saver(saver_params)

    
def sess_init(session=None):
    if session is None: session = tf.get_default_session()
    session.run(tf.global_variables_initializer())
    
    
def sess_save(session=None, path='chk/tf.tfn'):
    if session is None: session = tf.get_default_session()
    saver = sess_saver()
    saver.save(session, path)

    
def sess_load(session=None, path='chk/tf.tfn'):
    if session is None: session = tf.get_default_session()
    saver = sess_saver()
    import os.path
    if os.path.isfile(path): 
        saver.restore(session, path)
    else: 
        print('load failed because file (%s) does not exist' % path)
        #init(session)

    
#%%

relu = tf.nn.relu

softmax = tf.nn.softmax

sigmoid = tf.nn.sigmoid


def maxpool(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME')
    
def conv(x, w, b=None, stride=1, pool=0, relu=False):
    res = tf.nn.conv2d(x, w, strides=[1,stride,stride,1],padding='SAME')
    if b is not None: res = tf.nn.bias_add(res, b)
    if pool>1: res = maxpool(res, pool)
    if relu: res = tf.nn.relu(res)
    return res

def fwd(x, w, b=None, relu=False):
    res = tf.matmul(x,w)
    if b is not None: res = tf.nn.bias_add(res, b)
    if relu: res = tf.nn.relu(res)
    return res

def s2img(x, width, height, channel):
    return tf.reshape(x, [-1, width, height, channel])

def s2vec(x):
    shape = x.get_shape()
    n = 1
    for i in shape[1:]: n *= i.value
    return tf.reshape(x, [-1, n])

#%%
def loss_reg(regfactor=1, prefix='w', suffix=''):
    res = 0
    for i in tf.trainable_variables():
        name = i.name[0:i.name.find(':')]
        if name.startswith(prefix) and name.endswith(suffix):
            res += regfactor*tf.reduce_sum(tf.square(i))
    return res
    
def loss_mse(yhat, y):
    return tf.reduce_mean(tf.squared_difference(yhat, y))

def loss_softmax(yhat_logits, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yhat_logits, y))

    
def loss_policy(policy, action_mask, advantage, max_advantage=None):
    phatm = tf.reduce_sum(tf.mul(policy, action_mask), reduction_indices=[1])
    adv = advantage
    if max_advantage is not None and max_advantage > 0:
        adv = tf.minimum(adv, max_advantage)
        adv = tf.maximum(adv, -max_advantage)
    eligibility = tf.mul(tf.log(phatm), adv)
    loss = tf.neg(tf.reduce_mean(eligibility))
    return loss

    
def loss_qy(q, action_mask, y):
    q_a = tf.reduce_sum(tf.mul(q, action_mask), reduction_indices=[1])
    return loss_mse(q_a, y)

    
def loss_learn(loss, learning_rate=0.01, optimizer='adam'):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    
def eval_accuracy(yhat, y):
    correct = tf.equal(tf.arg_max(yhat,1), tf.arg_max(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    return accuracy

#%%

def policy_act(prob, rand_policy=True, take_chance=None):
    n = len(prob)
    if take_chance is not None and np.random.random() < take_chance:
        return np.random.choice(n)
    if rand_policy:
        return np.random.choice(n, p=prob)
    return prob.argmax()