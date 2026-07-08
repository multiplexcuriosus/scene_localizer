from setuptools import find_packages, setup
from glob import glob

package_name = 'scene_localizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/table_marker_layout.yaml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jau',
    maintainer_email='jau.gretler@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scene_localizer = scene_localizer.scene_localizer_node:main',
            'scene_localizer_debug = scene_localizer.scene_localizer_debug_node:main',
        ],
    },
)
