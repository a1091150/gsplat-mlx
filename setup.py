from setuptools import setup
from mlx import extension
import os

PACKAGE_NAME = "gsplat_core"


if __name__ == "__main__":
    cmake_args = os.environ.get("CMAKE_ARGS", "")
    required_args = [
        "-DGSPLAT_BUILD_PYTHON=ON",
        "-DGSPLAT_BUILD_TEST=OFF",
        "-DGSPLAT_BUILD_METAL=ON",
        "-DGSPLAT_USE_MLX_MODULE_CMAKE_DIR=OFF",
    ]
    for arg in required_args:
        if arg not in cmake_args:
            cmake_args = f"{cmake_args} {arg}".strip()
    os.environ["CMAKE_ARGS"] = cmake_args

    setup(
        name=PACKAGE_NAME,
        version="0.0.1",
        description="gsplat MLX custom extension package",
        ext_modules=[
            extension.CMakeExtension(f"{PACKAGE_NAME}._gsplat_core")
        ],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=[PACKAGE_NAME],
        package_dir={"": "python_package"},
        package_data={
            PACKAGE_NAME: ["*.so", "*.dylib", "*.metallib"],
        },
        include_package_data=True,
        zip_safe=False,
        python_requires=">=3.11",
    )
