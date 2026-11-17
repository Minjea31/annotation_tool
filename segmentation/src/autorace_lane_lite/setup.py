from setuptools import find_packages, setup

setup(
    name='autorace_lane_lite', version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/autorace_lane_lite']),
        ('share/autorace_lane_lite', ['package.xml']),
        ('share/autorace_lane_lite/config', ['config/lane.yaml']),
        ('share/autorace_lane_lite/launch', ['launch/lane.launch.py']),
    ],
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=True,
    maintainer='Minjea31', maintainer_email='kimminjea31@naver.com',
    description='MobileNetV3-Small U-Net lane segmentation with shared BEV tracking and OpenCV fallback.',
    license='Apache-2.0', entry_points={'console_scripts': ['lane_node = autorace_lane_lite.node:main']},
)
