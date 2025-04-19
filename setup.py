from setuptools import setup, find_packages


setup(
    name="dyeles",  # 包名
    version="0.1.0",  # 版本号
    description="A simple tool to calculate DyeLikeness Scores from SMILES",  # 简要描述
    author="Silong Zhai",  # 作者
    packages=find_packages(),  # 自动查找
    install_requires=[],
    include_package_data=True,
    python_requires=">=3.9",
    entry_points={"console_scripts": ["dyeles-score = dyeles.cli:main"]},
)
