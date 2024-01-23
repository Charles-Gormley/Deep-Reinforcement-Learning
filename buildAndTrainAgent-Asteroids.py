import gym
import ale_py
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils.common import function
from tf_agents.utils import common
from tf_agents.environments.tf_py_environment import TFPyEnvironment

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import PIL
from PIL import Image

import os
import logging

# Set up logging
mpl.rc('animation', html='jshtml')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')


n_iter = 10


logging.info("Importing necessary modules for the environment and agent.")

# Set up the Gym environment for StarGunner
max_episode_steps = 27000
env_name = 'AsteroidsNoFrameskip-v4'  # Use the appropriate environment name
logging.info(f"Loading environment: {env_name}")
env = suite_gym.load(env_name)

env = suite_atari.load(
    env_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessing, FrameStack4]
)
logging.info("Environment loaded and wrapped with Atari preprocessing and frame stacking.")

# Convert to a TensorFlow environment
tf_env = TFPyEnvironment(env)
logging.info("Converted the Gym environment to TensorFlow environment.")

# Create a Q-Network
fc_layer_params = (100,)  # Define the size of the neural network layers
preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]  # TODO: Hyperparameterize
fc_layer_params = [512]

q_net = q_network.QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params
)
logging.info("Q-Network created with specified layers and parameters.")

################## Agent Initialization ##################
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=2.5e-4,
    rho=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True
)
logging.debug("Optimizer initialized with RMSprop parameters.")

train_step = tf.Variable(0)
update_period = 4
epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0,
    decay_steps=250000 // update_period,
    end_learning_rate=0.01
)

agent = DqnAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    target_update_period=2000,  # <=> 32,000 ALE frames
    td_errors_loss_fn=tf.keras.losses.Huber(reduction="none"),
    gamma=0.99,  # discount factor
    train_step_counter=train_step,
    epsilon_greedy=lambda: epsilon_fn(train_step)
)

agent.initialize()
logging.info("DQN agent initialized and ready to be trained.")

################## Training ##################
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec, 
                                                               batch_size=tf_env.batch_size, 
                                                               max_length=1000)
replay_buffer_observer = replay_buffer.add_batch

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]
log_metrics(train_metrics)

collect_driver = DynamicStepDriver(tf_env,
                                   agent.collect_policy,
                                   observers=[replay_buffer_observer] + train_metrics,
                                   num_steps=update_period)

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total

    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")


initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
init_driver = DynamicStepDriver(tf_env, initial_collect_policy, observers=[replay_buffer.add_batch, ShowProgress(20000)], num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

dataset = replay_buffer.as_dataset(sample_batch_size=64,
                                   num_steps=2,
                                   num_parallel_calls=3).prefetch(3)

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

frames = []
def save_frames(trajectory):
    global frames
    frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))
def create_gif(frames, filename, fps=30):
    # Convert numpy arrays to PIL images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    pil_frames[0].save(filename,
                       format='GIF',
                       append_images=pil_frames[1:],
                       save_all=True,
                       duration=1000/fps,  # Duration per frame in milliseconds
                       loop=0)
    
def save_checkpoint(iteration):
    checkpoint_dir = f'./checkpoints/asteroids_{iteration}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, agent=agent, global_step=train_step)
    train_checkpointer.save(train_step)

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
        iteration, train_loss.loss.numpy()), end="")

        save_checkpoint(iteration)

        # Run the watch driver to collect frames
        watch_driver = DynamicStepDriver(tf_env,
                                        agent.policy,
                                        observers=[save_frames, ShowProgress(5000)],
                                        num_steps=5000)
        final_time_step, final_policy_state = watch_driver.run()

        create_gif(frames, f'StarGunner-{iteration}.gif', fps=30)

        if iteration % 1000 == 0:
            log_metrics(train_metrics)
train_agent(n_iterations=n_iter)




################## Collection ##################

# # Run the watch driver to collect frames
# watch_driver = DynamicStepDriver(tf_env,
#                                  agent.policy,
#                                  observers=[save_frames, ShowProgress(1000)],
#                                  num_steps=10000)
# final_time_step, final_policy_state = watch_driver.run()

################## Save as GIF ##################

