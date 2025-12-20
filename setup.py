from setuptools import setup, find_packages

setup(
    name="slm-lookback-detector",
    version="0.1.0",
    description="Lookback-Lens-style backdoor and hallucination detection for Small LLMs",
    author="Research",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "click>=8.1.0",
        "tqdm>=4.66.0",
    ],
    entry_points={
        "console_scripts": [
            "slm-detect=cli:main",
        ],
    },
)
