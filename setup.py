from setuptools import find_packages, setup

setup(
    name="llmeyesim",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'llmeyesim=LLMEyeSim.cli:main',  # This makes the 'llmeyesim' command available
        ],
    },
    python_requires='>=3.10',  # Adjust based on your needs
    author="Wenxiao",
    description="LLM Eye Simulation Project",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)