"""Setup for lenia_field package."""

from setuptools import setup, find_packages

setup(
    name="lenia_field",
    version="0.1.0",
    description="Real-time Lenia morphogen field with external stimulus injection",
    author="Charles",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20.0",
        "pyzmq>=22.0.0",
        "msgpack>=1.0.0",
        "msgpack-numpy>=0.4.8",
    ],
    extras_require={
        "gpu": ["jax[cuda12]>=0.4.0"],  # For CUDA support
        "viz": ["opencv-python>=4.5.0", "matplotlib>=3.5.0"],
    },
    entry_points={
        "console_scripts": [
            "lenia-server=lenia_field.server:main",
        ],
    },
)
