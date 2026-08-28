from setuptools import find_packages, setup

package_name = 'sky_mapper'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/sky_mapper.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='primordia@live.com',
    description='Constructs a calibrated equirectangular sky panorama from an upward-facing camera.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sky_mapper_node = sky_mapper.sky_mapper_node:main',
        ],
    },
)
