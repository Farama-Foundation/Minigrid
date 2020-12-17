from setuptools import setup

setup(
    name='gym_minigrid',
    version='1.0.2',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=['gym_minigrid', 'gym_minigrid.envs'],
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
