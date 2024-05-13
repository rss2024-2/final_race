from setuptools import setup

package_name = 'final_race'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='racecar',
    maintainer_email='mohammedehab200218@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'line_detector = final_race.line_following.line_detector:main',
            'homography_transformer = final_race.line_following.homography_transformer:main',
            'hough_follower = final_race.line_following.hough_follower:main',
            'pure_pursuit = final_race.pure_pursuit:main'
        ],
    },
)
