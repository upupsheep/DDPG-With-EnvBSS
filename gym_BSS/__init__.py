from gym.envs.registration import register

register(
    id='BSSEnv-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'use_test_data': False
    }
)

register(
    id='BSSEnvTest-v0',
    entry_point='gym_BSS.envs:BSSEnv',
    kwargs={
        'use_test_data': True
    }
)
