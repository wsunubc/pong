# This module define the Pong AI

# Import 
import tensorflow as tf
import numpy as np
import gym

# Import my utility procedures for tensorflow
from tfsh import get_weight, get_bias, fwd, softmax, policy_act
from tfsh import loss_qy, loss_reg, loss_learn
from tfsh import sess_save, sess_load, sess_init


# Define hyper-parameters for the game
# Valid actions for the game: 1-Stay, 2-Up, 3-Down
action_map = [1, 2, 3]
NActions = len(action_map)
action_mask = np.eye(NActions)

NHidden = 200
LearningRate = 1e-3
LearningRateDecay = 0.99
RewardDiscount = 0.99

NObserve = 0
NMem = 20000
NMiniBatch = 500

env = gym.make('Pong-v0')
def step(action):
    return env.step(action_map[action])



#Image Processing    
# img_prepro removes irrelevant info from the game image to speed up learning
# adapted from github.com/karpathy/
def img_prepro(img):
    """ prepro 210x160x3 uint8 frame into 6400 float32 vector """
    if img is None: return None
    img = img[35:195] # crop
    img = img[::2,::2,0]  # one channel seems to be enough here
    img[img == 144] = 0 # erase background (background type 1)
    img[img == 109] = 0 # erase background (background type 2)
    img[img != 0] = 1 # everything else (paddles, ball) just set to 1
    return img.astype(np.float32).ravel()

    
def img_combine(imgs):
    return imgs[1] - imgs[0]


# Define the policy gradient neural network in tensorflow
# input layer
x = tf.placeholder(tf.float32, [None, 6400], name='x') #State - Past two screenshots
v = tf.placeholder(tf.float32, [None, 1], name='v') # Reward
m = tf.placeholder(tf.float32, [None, NActions], name='m') # Action Mask
# kp = tf.placeholder(tf.float32)

# weights
w1 = get_weight('w1', [6400, NHidden])
w = get_weight('w', [NHidden, NActions])


# hidden and output layer
# output layer is the probability for taking each action (i.e., a policy)
h1   = fwd(x , w1, None, relu=True)
logphat = fwd(h1, w,  None,  relu=False)
phat = softmax(logphat) 

# Normalize reward signal
# increase probability of actions that leads to relatively higher reward.
v_mean, v_var = tf.nn.moments(v, [0])
v_n = (v-v_mean)/tf.sqrt(v_var + 1e-6)

loss = tf.nn.l2_loss(m-phat)
optimizer = tf.train.RMSPropOptimizer(LearningRate,decay=LearningRateDecay)
grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables(), grad_loss=v_n)
trainop = optimizer.apply_gradients(grads)


# session
session = tf.Session()
run = session.run
sess_init(session)

def save():
    sess_save(session)

def load():
    sess_load(session)
    
#%%

def calc_action(cstate, take_chance = 0):
    prob = run(phat, feed_dict={x:[cstate]})[0]
    return policy_act(prob, take_chance=take_chance)
    

def train(mstate, maction, mreward):
    l, _ = run([loss, trainop], feed_dict={x:mstate, v:mreward, m:maction})
    return l

def run_episode(take_chance=0, learn=False, pause=0):
    from collections import deque
    mstate = []
    maction = []
    mreward = []
    mobs = []
    
    cimg = img_prepro(env.reset())
    rimg = deque(maxlen=2)
    rimg.append(cimg)
    rimg.append(cimg)
    
    cdone = False
    total_steps = 0
    total_reward = 0
    
    while not cdone:
        if pause > 0:
            import time
            time.sleep(pause)
            env.render()
            
        cstate = img_combine(rimg)
        caction = calc_action(cstate, take_chance=take_chance)
        cobs, creward, cdone, _ = step(caction)
        cobs = img_prepro(cobs)
        rimg.append(cobs)
        
        mstate.append(cstate)
        maction.append(action_mask[caction])
        mreward.append(float(creward))
        mobs.append(cobs)

        total_steps += 1
        total_reward += creward

    l = None
    if learn:
    	while len(reward)>0 and reward[-1]==0:
            del reward[-1], mstate[-1], maction[-1], mobs[-1]
        n = len(mreward)
        for i in reversed(range(n-1)):
            if mreward[i]==0: mreward[i]=RewardDiscount*mreward[i+1]
        mreward = np.reshape(mreward, (-1,1))
        l = train(mstate, maction, mreward)
        
    return total_reward, total_steps, l


# run training for a number of episodes
def run_train(episodes=100, take_chance = 0, take_chance_decay = 0.99):
    load()
    eval_size = 10
    max_avg_score = None
    cur_avg_score = 0
    avg_steps = 0
    for i in range(episodes):
        cur_score, cur_steps, l = run_episode(take_chance=take_chance, learn=True)
        take_chance*= take_chance_decay
        cur_avg_score += cur_score
        avg_steps += cur_steps
        if (i+1)%eval_size == 0:
            cur_avg_score /= eval_size
            avg_steps /= eval_size
            if max_avg_score is None: max_avg_score = cur_avg_score
            print('Episode', i+1, ': score=', cur_avg_score, '(%+.2f)'%(cur_avg_score-max_avg_score), 'steps=', avg_steps, 'loss=', l)
            if cur_avg_score >= max_avg_score:
                max_avg_score = cur_avg_score
                save()
            cur_avg_score = 0
            avg_steps = 0
        
    print('Training completed. score=', max_avg_score)
    return max_avg_score


# run evaluation
def run_evaluation(trials=100, rand_policy=False, take_chance=0):
    load()
    import matplotlib.pyplot as plt
    scores = []
    for _ in range(trials):
        cur_score, cur_steps = run_episode(take_chance=take_chance, learn=False)
        scores.append(cur_score)
    plt.hist(scores)
    plt.show()
    min_score, max_score, avg_score = int(min(scores)), int(max(scores)), int(round(sum(scores)/len(scores)))
    print('Min Score is', min_score)
    print('Max Score is', max_score)
    print('Avg Score is', avg_score)
    return min_score, max_score, avg_score

