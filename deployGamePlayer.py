import argparse
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils import common
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from PIL import Image
import numpy as np
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments import suite_atari
from tf_agents.environments.atari_wrappers import FrameStack4

import logging

def create_gif(frames, filename, fps=30):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(filename, format='GIF', append_images=pil_frames[1:], save_all=True, duration=1000/fps, loop=0)

def load_agent_and_play_game(checkpoint_dir, env_name, num_frames=1000, fps=30):
    max_episode_steps = 27000
    env_name = 'AsteroidsNoFrameskip-v4'  # Use the appropriate environment name
    logging.info(f"Loading environment: {env_name}")
    env = suite_gym.load(env_name)

    env = suite_atari.load(
        env_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
)
    tf_env = tf_py_environment.TFPyEnvironment(env)

    sample_observation = env.reset().observation
    print("Shape of observation from the environment:", sample_observation.shape)


    preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = q_network.QNetwork(tf_env.observation_spec(), tf_env.action_spec(), preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )
    
    optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=2.5e-4,
    rho=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True
    )
    train_step = tf.Variable(0)
    update_period = 3
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

    train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, agent=agent, global_step=train_step)
    status = train_checkpointer.initialize_or_restore()
    status.expect_partial()

    frames = []
    policy_state = agent.policy.get_initial_state(tf_env.batch_size)
    time_step = tf_env.reset()

    for _ in range(num_frames):
        policy_step = agent.policy.action(time_step, policy_state)
        time_step = tf_env.step(policy_step.action)
        frames.append(tf_env.pyenv.envs[0].render(mode="rgb_array"))

    create_gif(frames, 'agent_performance.gif', fps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load saved policy and play game')
    parser.add_argument('--checkpoint_path', type=str, help='Path to the policy/model/checkpoint')
    parser.add_argument('--env_name', type=str, default='AsteroidsNoFrameskip-v4', help='Gym environment name')
    parser.add_argument('--num_frames', type=int, default=1000, help='Number of frames to play and record')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second for the output GIF')
    args = parser.parse_args()


    num_frames = 300
    fps = 30
    checkpoint_num = 8
    env_name = 'AsteroidsNoFrameskip-v4'
    checkpoint_path = f'./checkpoints/asteroids_{checkpoint_num}'

    load_agent_and_play_game(checkpoint_path, env_name, num_frames, fps)
