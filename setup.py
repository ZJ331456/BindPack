from setuptools import setup, find_packages

setup(
    name="bindpack",
    version="0.1.0",
    description="BindPack: source-bound dependency execution and evidence package selection for multi-hop RAG",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={"index-quality": ["xgrammar==0.1.18"]},
)
