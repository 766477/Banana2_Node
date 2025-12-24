from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# 这里填你要加密的文件名（不需要 .py 后缀）
file_names = ["Simple_Banana_Node"]

extensions = [
    Extension(
        name=name,
        sources=[f"{name}.py"],
    )
    for name in file_names
]

setup(
    name="Banana_Simple_Node",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)