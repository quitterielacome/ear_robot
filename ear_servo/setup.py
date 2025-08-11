from setuptools import setup

package_name = 'ear_servo'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', ['config/default.yaml']),
        ('share/' + package_name + '/launch', ['launch/bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Quitterie',
    maintainer_email='you@example.com',
    description='KLT+RANSAC tracking and image-based pose servo for xArm',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ear_servo_pose = ear_servo.ear_servo_pose_node:main',
            'ear_servo_velo = ear_servo.ear_servo_velo_node:main',
            'metrics_logger = ear_servo.metrics_logger:main',
        ],
    },
)
