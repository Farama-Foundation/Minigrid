from setuptools import find_packages, setup

with open("README.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        if header_count < 2:
            long_description += line
        else:
            break

# pytest is pinned to 7.0.1 as this is last version for python 3.6
extras = {"testing": ["pytest==7.0.1"]}

setup(
    name="MiniGrid",
    author="Farama Foundation",
    author_email="jkterry@farama.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    version="2.0.0",
    keywords="memory, environment, agent, rl, gymnasium",
    url="https://github.com/Farama-Foundation/MiniGrid",
    description="Minimalistic gridworld reinforcement learning environments",
    extras_require=extras,
    packages=[package for package in find_packages() if package.startswith("minigrid")],
    entry_points={
        "gymnasium.envs": ["__root__ = minigrid.__init__:register_minigrid_envs"]
    },
    license="Apache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "gymnasium>=0.26",
        "numpy>=1.18.0",
        "matplotlib>=3.0",
    ],
    python_requires=">=3.7",
    tests_require=extras["testing"],
)
