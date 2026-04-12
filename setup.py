import os
from glob import glob

from setuptools import find_packages, setup

package_name = "px4_slam"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Graham Clifford",
    maintainer_email="gjcliff@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "gtsam = px4_slam.gtsam:main",
            "match_points = px4_slam.match_points:main",
            "optical_flow = px4_slam.optical_flow:main",
            "super_flow = px4_slam.super_flow:main",
        ],
    },
)
