from setuptools import find_packages, setup

package_name = 'celestial_localizer'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/celestial_localizer.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='james',
    maintainer_email='primordia@live.com',
    description='Estimates robot latitude/longitude/heading from celestial observations.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'celestial_localizer_node = celestial_localizer.celestial_localizer_node:main',
        ],
    },
)
