from setuptools import setup

package_name = 'trajectory_planner'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    description='Trajectory planner and executor for XArm 7',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tracking_node = trajectory_planner.tracking_node:main',
            'trajectory_server = trajectory_planner.trajectory_server:main',
        ],
    },
)
