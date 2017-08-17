# AI for Pong

## Overview


[![AI that plays Pong](https://img.youtube.com/vi/aEM6rZ9KWeQ/0.jpg)](https://www.youtube.com/watch?v=aEM6rZ9KWeQ)

* AI that plays the [Atari Pong game](https://en.wikipedia.org/wiki/Pong).
* Trained with reinforcement learning.
* Outperforms the original Atari Pong AI.
* Implemented based on the blog post of [Andrej Karpathy](http://karpathy.github.io/2016/05/31/rl/)

## Prerequisites

* Anaconda 4.4 with Python 3.6
* Tensorflow v1.2
* OpenAI GYM - complete install
  - `pip install 'gym[all]'`

## Run

* `cd` to the directory where `main.py` and `chk` folder are located.
* Run command: `/opt/anaconda3/bin/python main.py`
  - Suppose you have installed anaconda 3 at `/opt/anaconda3`

## Future Work
* Combine supervised learning with reinforcement learning to speedup training.
* Learn and reimplement using the [A3C LSTM method](https://github.com/dgriff777/rl_a3c_pytorch) to increase AI performance.

