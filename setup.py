from setuptools import setup, find_packages

# Helper to load requirements from requirements.txt
def load_requirements(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="patho_llama",
    version="0.1.0",
    author="Bryan Cardenas",
    description="A multimodal extension of LLaMA for pathology tasks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=load_requirements("requirements.txt"),
    python_requires=">=3.8",
)
