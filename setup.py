from setuptools import setup
# from distutils.extension import Extension
# from Cython.Distutils import build_ext
# import numpy


setup(
    name="tov_solver",
    version="1.0",
    author="Maximilian Jacobi",
    author_email="mjacobi@theorie.ikp.physik.tu-darmstadt.de",
    packages=["tov_solver"],
    license='MIT',
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "alpyne @ git+https://github.com/fguercilena/alpyne@master",
        "tabulatedEOS @ git+https://github.com/max-jacobi/tabulatedEOS@master",
    ],
    python_requires='>=3.7',
    description="Routines to solve the TOV equations given a tabulated EOS",
)
