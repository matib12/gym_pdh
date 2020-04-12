from gym.envs.registration import register

register(
    id='pdh-v0',
    entry_point='gym_pdh.envs:PdhEnv',
    max_episode_steps=600,
    #reward_threshold=25.0,
)
