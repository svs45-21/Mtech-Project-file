from setuptools import find_packages, setup

setup(
    name="spark-assistant",
    version="1.0.0",
    description="SPARK Intelligent Personalized AI Operating Companion",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "PyQt6==6.7.1",
        "pydantic==2.9.2",
        "cryptography==43.0.1",
        "psutil==6.0.0",
        "SpeechRecognition==3.10.4",
        "pyttsx3==2.98",
        "requests==2.32.3",
        "PyAudio==0.2.14",
    ],
    python_requires=">=3.11",
)
