from setuptools import setup

setup(
    name="kinder_garten",
    version="0.0.1",
    install_requires=["gym",
                      "pybullet",
                      "stable-baselines3[extra]",
                      "scikit-learn "],
    packages=['kinder_garten'],
)
