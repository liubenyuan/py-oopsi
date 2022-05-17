
import setuptools


base_packages = [
    "numpy",
    "scipy",
    "matplotlib",
]


setuptools.setup(
    name="oopsi",
    version="0.1.0",
    author="liubenyuan",
    author_email="liubenyuan@gmail.com",
    packages=["oopsi"],
    install_requires=base_packages,
    python_requires=">=3.8",
)
