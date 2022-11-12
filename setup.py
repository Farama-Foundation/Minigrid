"""Setups up the Minigrid module."""

from __future__ import annotations

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
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
    return header_count, long_description


def get_version():
    """Gets the minigrid version."""
    path = "minigrid/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


def get_requirements():
    """Gets the description from the readme."""
    with open("requirements.txt") as reqs_file:
        reqs = reqs_file.readlines()
    return reqs


def get_tests_requirements():
    """Gets the description from the readme."""
    with open("test_requirements.txt") as test_reqs_file:
        test_reqs = test_reqs_file.readlines()
    return test_reqs


# pytest is pinned to 7.0.1 as this is last version for python 3.6
extras = {"testing": get_tests_requirements()}

version = get_version()
header_count, long_description = get_description()

setup(
    name="Minigrid",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Minimalistic gridworld reinforcement learning environments",
    url="https://minigrid.farama.org/",
    license="Apache",
    license_files=("LICENSE",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Memory, Environment, Agent, RL, Gymnasium"],
    python_requires=">=3.7, <3.11",
    packages=[package for package in find_packages() if package.startswith("minigrid")],
    include_package_data=True,
    install_requires=get_requirements(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,
    entry_points={
        "gymnasium.envs": ["__root__ = minigrid.__init__:register_minigrid_envs"]
    },
    tests_require=extras["testing"],
)
