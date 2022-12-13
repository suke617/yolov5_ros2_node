import os
from glob import glob
from setuptools import setup

package_name = 'yolov5_ros2'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jeffrey Too Chuan TAN',
    maintainer_email='ai-robot-book@googlegroups.com',
    description='YOLOv5 applications with ROS 2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = yolov5_ros2.detector:main',
            'object_detection = yolov5_ros2.yolov5_detector:main',
        ],
    },
)
