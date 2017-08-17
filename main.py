#%%
import time
import PongPNN as pnn


#%% Video Demo
pnn.load()
pnn.run_episode(take_chance=0, learn=False, pause=0.02)


"""
The code in this comment demonstrates how the neural network has been trained.

#%% General training
tm0 = time.time()
for i in range(2000):
    pnn.run_train(episodes=200, take_chance=0, take_chance_decay=0.97)
    print ('%d seconds elapsed since training started.'%(time.time() - tm0))
    pnn.save()


#%% Evaluation
pnn.run_evaluation()


#%% Manual save
pnn.save()
"""


