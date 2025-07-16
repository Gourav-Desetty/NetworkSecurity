from setuptools import setup, find_packages
from typing import List

Hyphen_e_dot = '-e .'
def get_requirements(file_path)->List[str]:
    requirements = []
    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n", "") for req in requirements]

            if Hyphen_e_dot in requirements:
                requirements.remove(Hyphen_e_dot)

        return requirements
    except:
        print("requirements.txt file not found")
        return []


setup(
    name = "NetworkSecurity",
    version= "0.0.1",
    author= "Gourav Desetty",
    author_email= "dg.kgp10@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)