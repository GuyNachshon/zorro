import os
from setuptools import setup
 
 
VERSION="2.0.0"
OLD_PACKAGE_NAME="django-rest-multitokenauth"
NEW_PACKAGE_NAME="drf-multitokenauth"
 
 
def get_description():
    return "{old_package_name} is deprecated now. Use {new_package_name} instead.".format(
        old_package_name=OLD_PACKAGE_NAME,
        new_package_name=NEW_PACKAGE_NAME,
    )
 
 
setup(
    name=OLD_PACKAGE_NAME,
    description=get_description(),
    long_description=get_description(),
    long_description_content_type="text/markdown",
    version=VERSION,
    install_requires=[
        NEW_PACKAGE_NAME,
    ],
    classifiers=["Development Status :: 7 - Inactive"],
)
