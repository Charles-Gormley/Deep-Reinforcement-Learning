import tensorflow as tf
from tf_agents.environments import suite_gym, suite_atari
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.agents.dqn.dqn_agent import DqnAgent
from tf_agents.utils import common
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.metrics import tf_metrics
import numpy as np
import logging

def resume_training(checkpoint_dir, env_name, num_iterations=100):
    # Environment setup
    max_episode_steps = 27000
    env = suite_atari.load(
        env_name,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=[AtariPreprocessing, FrameStack4]
    )
    tf_env = tf_py_environment.TFPyEnvironment(env)

    # Q-Network
    preprocessing_layer = tf.keras.layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255.)
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    q_net = q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params
    )

    # DQN Agent
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4,
        rho=0.95,
        momentum=0.0,
        epsilon=0.00001,
        centered=True
    )
    train_step = tf.Variable(0)

    agent = DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        train_step_counter=train_step
    )
    agent.initialize()

    # Load checkpoint
    train_checkpointer = common.Checkpointer(ckpt_dir=checkpoint_dir, agent=agent, global_step=train_step)
    train_checkpointer.initialize_or_restore()

    # Replay Buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=1000
    )

    # Metrics and Driver
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]



    # Initial Data Collection
    initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(), tf_env.action_spec())
    init_driver = DynamicStepDriver(
        tf_env,
        initial_collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=100  # Adjust as needed
    )
    final_time_step, final_policy_state = init_driver.run()

    # Convert the replay buffer to a dataset
    dataset = replay_buffer.as_dataset(
        sample_batch_size=64,
        num_steps=2,
        num_parallel_calls=3,
        single_deterministic_pass=False
    ).prefetch(3)

    collect_driver = DynamicStepDriver(
        tf_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=1  # Collect 1 step for each environment step
    )


    # Training loop
    iterator = iter(dataset)
    for _ in range(num_iterations):
        # Collect a few steps and save to the replay buffer
        
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        # Log training information
        logging.info(f"Step: {train_step.numpy()}, Loss: {train_loss}")

if __name__ == '__main__':
    resume_training(
        checkpoint_dir='./checkpoints/asteroids',
        env_name='AsteroidsNoFrameskip-v4',
        num_iterations=10000
    )
