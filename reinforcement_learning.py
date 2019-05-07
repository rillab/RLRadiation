import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf
from collections import namedtuple
import importlib as imp
import simulation as sm
import matplotlib.pyplot as plt
import time
from tensorflow.contrib import rnn

#Global variables
WORLD_SIZE = 10
VALID_ACTIONS = [0,1,2,3]


if "../" not in sys.path:
    sys.path.append("../")

def generateBuildings():
    """
    Generate buildings in the simulation environment
    """
    wall_side = np.random.randint(4)
    wall_position = np.random.randint(int(0.3*WORLD_SIZE),int(0.7*WORLD_SIZE))
    length = np.random.randint(int(0.5*WORLD_SIZE),int(WORLD_SIZE))
    
    if wall_side==0:
        bd = set((wall_position,i) for i in range(1, length))
    elif wall_side==1:
        bd = set((wall_position,WORLD_SIZE+1-i) for i in range(1, length))
    elif wall_side==2:
        bd = set((i, wall_position) for i in range(1, length))
    else:
        bd = set((WORLD_SIZE+1-i, wall_position) for i in range(1, length))
    
    return [bd]
    

class ModelParametersCopier():
    """
    Copy model parameters of one estimator to another.
    """
    
    def __init__(self, estimator1, estimator2):
        """
        Defines copy-work operation graph.  
        Args:
          estimator1: Estimator to copy the paramters from
          estimator2: Estimator to copy the parameters to
        """
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        """
        Makes copy.
        Args:
            sess: Tensorflow session instance
        """
        sess.run(self.update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

class Estimator_CNN():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """
        # input data type (batch, WORLD_SIZE,WORLD_SIZE, features); feature:[measurement number, radiation level, current position].
        self.X_pl = tf.placeholder(tf.float32, [None,WORLD_SIZE+2,WORLD_SIZE+2,3], name="X")        
        
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")
        
        batch_size = tf.shape(self.X_pl)[0]
        
        
        # ----------------------CNN---------
        conv1 = tf.layers.conv2d(
            inputs=self.X_pl,
            filters=8,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
          # Convolutional Layer #2 and Pooling Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flatten = tf.contrib.layers.flatten(conv2)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        #regularizer = None
        fc1 = tf.contrib.layers.fully_connected(pool2_flatten, 16, weights_regularizer=regularizer, activation_fn=tf.nn.sigmoid) 
        fc2 = tf.contrib.layers.fully_connected(fc1, 8, weights_regularizer=regularizer, activation_fn=tf.nn.sigmoid) 
        self.predictions = tf.contrib.layers.fully_connected(fc2, len(VALID_ACTIONS), weights_regularizer=regularizer, activation_fn=None)

        
        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.losses.huber_loss(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, WORLD_SIZE, WORLD_SIZE, 3]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated 
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, WORLD_SIZE, WORLD_SIZE, 3]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss
    
def main():
    imp.reload(sm)
    env = sm.RadEnv2D_New(WORLD_SIZE, WORLD_SIZE)
    test_env = sm.RadEnv2D_New(WORLD_SIZE,WORLD_SIZE)

    tf.reset_default_graph()

    # Where we save our checkpoints and graphs
    experiment_dir = os.path.abspath("./experiments/DCQN_23")

    # Create a glboal step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Create estimators
    q_estimator = Estimator_CNN(scope="q_estimator", summaries_dir=experiment_dir)
    target_estimator = Estimator_CNN(scope="target_q")

    sess = tf.InteractiveSession()
    q_estimator=q_estimator
    target_estimator=target_estimator
    experiment_dir=experiment_dir
    num_episodes=10000000
    replay_memory_size=500000*5 #base: 500k
    replay_memory_init_size=1500
    update_target_estimator_every=10
    epsilon_start=1
    epsilon_end=0.1
    epsilon_decay_steps=1000000
    discount_factor=0.9
    batch_size=32
    maximum_steps = 100
    sess.run(tf.global_variables_initializer())

    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # Make model copier object
    estimator_copy = ModelParametersCopier(q_estimator, target_estimator)

    # Keeps track of useful statistics
    EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards"])
    stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    # For 'system/' summaries, usefull to check if currrent process looks healthy
    current_process = psutil.Process()

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    monitor_path = os.path.join(experiment_dir, "monitor")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(monitor_path):
        os.makedirs(monitor_path)

    saver = tf.train.Saver(max_to_keep=None)
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())

    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        q_estimator,
        len(VALID_ACTIONS))

    # Populate the replay memory with initial experience
    #-------------------------------------------------------------------------------------------------------------
    print("Populating replay memory...")
    
    state = env.resetCNN(generateBuildings(), current_location='random', source_location='random')

    tmp_t = 0
    for i in range(replay_memory_init_size):
        action_probs = policy(sess, state, epsilons[min(total_t, epsilon_decay_steps-1)])
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        next_state, reward, done = env.stepCNN(VALID_ACTIONS[action])
        replay_memory.append(Transition(state, action, reward, next_state, done))
        if done or tmp_t>maximum_steps:
            current_location=None
            source_location=None
            state = env.resetCNN(generateBuildings(), current_location='random', source_location='random')
            tmp_t = 0
        else:
            state = next_state
            tmp_t = tmp_t+1

    print("Start training ...")
    for i_episode in range(num_episodes):

        # Reset the environment
        state = env.resetCNN(generateBuildings(), current_location='random', source_location='random')
        loss = None

        #add test module:
        if i_episode %100 == 0:

            if i_episode % 500 == 0:
            # Save the current checkpoint
                saver.save(tf.get_default_session(), checkpoint_path+str(i_episode))

            mean_reward = []
            for kk in range(30):
                test_state = test_env.resetCNN(generateBuildings(), current_location='random', source_location='random')
                test_total_reward = 0
                test_steps = 0
                while test_steps <= maximum_steps:
                    # Take a step
                    test_action_probs = policy(sess, test_state, 0.05)
                    test_action = np.random.choice(np.arange(len(test_action_probs)), p=test_action_probs)
                    test_next_state, test_reward, test_done= test_env.stepCNN(VALID_ACTIONS[test_action])
                    test_total_reward = test_total_reward + test_reward

                    if test_done:
                        break
                    else:
                        test_state = test_next_state
                        test_steps = test_steps + 1

                mean_reward.append(test_total_reward)
            mean_reward = np.array(mean_reward)
            #test summary
            test_summary = tf.Summary()
            test_summary.value.add(simple_value=mean_reward.mean(), tag="episode/test_reward_mean")
            test_summary.value.add(simple_value=mean_reward.std(), tag="episode/test_reward_std")
            test_summary.value.add(simple_value=mean_reward.max(), tag="episode/test_reward_max")
            test_summary.value.add(simple_value=mean_reward.min(), tag="episode/test_reward_min")
            q_estimator.summary_writer.add_summary(test_summary, i_episode)


        # One step in the environment
        for t in itertools.count():

            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]

            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estimator_copy.make(sess)
                #print("\nCopied model parameters to target network.")

            # Print out which step we're on, useful for debugging.
            #print("\rStep {} ({}) @ Episode {}/{}, loss: {}".format(
            #        t, total_t, i_episode + 1, num_episodes, loss))
            #sys.stdout.flush()

            # Take a step
            action_probs = policy(sess, state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done= env.stepCNN(VALID_ACTIONS[action])

            # If our replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # Save transition to replay memory
            replay_memory.append(Transition(state, action, reward, next_state, done))   

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # Sample a minibatch from the replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate q values and targets
            # This is where Double Q-Learning comes in!
            q_values_next = q_estimator.predict(sess, next_states_batch)
            best_actions = np.argmax(q_values_next, axis=1)
            q_values_next_target = target_estimator.predict(sess, next_states_batch)
            targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * \
                discount_factor * q_values_next_target[np.arange(batch_size), best_actions]

            # Perform gradient descent update
            states_batch = np.array(states_batch)
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch)

            if done or t>maximum_steps:
                if i_episode % 20 == 0:
                    print('\nepisode {}: t is {}, reward is {}'.format(
                        i_episode, stats.episode_lengths[i_episode], stats.episode_rewards[i_episode]))
                    print(state[:,:,0].max(), state[:,:,1].max(), state[:,:,2].max()) #tmp for debug
                break
            
            state = next_state
            total_t += 1

        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=epsilon, tag="episode/epsilon")
        episode_summary.value.add(simple_value=stats.episode_rewards[i_episode], tag="episode/reward")
        episode_summary.value.add(simple_value=stats.episode_lengths[i_episode], tag="episode/length")
        episode_summary.value.add(simple_value=current_process.cpu_percent(), tag="system/cpu_usage_percent")
        episode_summary.value.add(simple_value=current_process.memory_percent(memtype="vms"), tag="system/v_memeory_usage_percent")
        q_estimator.summary_writer.add_summary(episode_summary, i_episode)
        q_estimator.summary_writer.flush()
        
if __name__ == "__main__":
    main()
