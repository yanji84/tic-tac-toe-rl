import tensorflow as tf
import numpy as np
from collections import deque
from rl.deep_q_network import DeepQNetwork
from game import Game

# initialize game env
env = Game()

# initialize tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
writer = tf.train.SummaryWriter("logs/value_network", sess.graph)

# prepare custom tensorboard summaries
episode_reward = tf.Variable(0.)
tf.scalar_summary("Last 100 Episodes Average Episode Reward", episode_reward)
summary_vars = [episode_reward]
summary_placeholders = [tf.placeholder("float") for i in range(len(summary_vars))]
summary_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]

# define policy neural network
state_dim   = 9
num_actions = 9
def value_network(states):
  W1 = tf.get_variable("W1", [state_dim, 256],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b1 = tf.get_variable("b1", [256],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

  W2 = tf.get_variable("W2", [256, 64],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  b2 = tf.get_variable("b2", [64],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

  Wo = tf.get_variable("Wo", [64, num_actions],
                       initializer=tf.random_normal_initializer(stddev=0.1))
  bo = tf.get_variable("bo", [num_actions],
                       initializer=tf.constant_initializer(0))

  p = tf.matmul(h2, Wo) + bo
  return p

summaries = tf.merge_all_summaries()
q_network = NeuralQLearner(sess,
                           optimizer,
                           value_network,
                           state_dim,
                           num_actions,
                           init_exp=0.6,         # initial exploration prob
                           final_exp=0.1,        # final exploration prob
                           anneal_steps=120000,  # N steps for annealing exploration
                           discount_factor=0.8)  # no need for discounting

# load checkpoint if there is any
saver = tf.train.Saver()
checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("successfully loaded checkpoint")

# how many episodes to train
training_episodes = 200000

# store episodes history
episode_history = deque(maxlen=100)

# start training
reward = 0.0
for i_episode in xrange(training_episodes):
  state = np.array(env.reset())
  for t in xrange(20):
    action = q_network.eGreedyAction(state[np.newaxis,:])
    next_state, reward, done = env.step(action)
    q_network.storeExperience(state, action, reward, next_state, done)
    q_network.updateModel()
    state = np.array(next_state)
    if done:
      episode_history.append(reward)
      break

  # print status every 100 episodes
  if i_episode % 100 == 0:
    mean_rewards = np.mean(episode_history)
    print("Episode {}".format(i_episode))
    print("Reward for this episode: {}".format(reward))
    print("Average reward for last 100 episodes: {}".format(mean_rewards))
    # update tensorboard
    sess.run(summary_ops[0], feed_dict = {summary_placeholders[0]:float(mean_rewards)})
    result = sess.run(summaries)
    writer.add_summary(result, i_episode)
    # save checkpoint
    saver.save(sess, "model/saved_network")