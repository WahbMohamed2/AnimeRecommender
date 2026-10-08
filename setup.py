from setuptools import setup, find_packages


def read_requirements(path="requirements.txt"):
    requirements = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if line:
                requirements.append(line)
    return requirements


setup(
    name="ANIME-RECOMMENDER",
    version="0.1",
    author="Sudhanshu",
    packages=find_packages(),
    install_requires=read_requirements(),
)
