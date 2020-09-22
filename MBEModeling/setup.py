from setuptools import find_packages, setup

setup(
    name="MBE",
    packages=find_packages(),
    include_package_data=True,
    version="0.1.0",
    description="Package to run MNIST batch size experiments",
    author="Matthew Almeida",
    license="MIT",
    install_requires=[
        "Click",
    ],
    entry_points="""
        [console_scripts]
        run=main:main
    """
)