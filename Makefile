.PHONY: install_python39
.DEFAULT_GOAL := image
UNAME := $(shell uname)
PYTHON_VERSION_MINOR:= $(wordlist 2,3,$(subst ., ,$(shell python3.9 --version 2>&1)))
POETRY_VERSION ?=1.1.13

IMAGE_MODEL ?= "resnet50"
BATCH_SIZE ?=64
NUM_IMAGES ?=100000
USE_GPU ?=1


define install_python39_mac
	@brew install python@3.9
endef

define install_python39_linux
	apt -y update
	apt -y install software-properties-common
	add-apt-repository ppa:deadsnakes/ppa -y
	DEBIAN_FRONTEND=noninteractive apt -y install python3.9
	apt -y install python3.9-dev
	apt -y install python3.9-distutils
endef

install_python39:
ifeq ("$(PYTHON_VERSION_MINOR)", "3 9")
	@echo "Python 3.9 found. Installing ..."
else
ifeq ($(UNAME),Darwin)
	echo "Installing Python 3.9 on MAC OS"
	$(call install_python39_mac)
endif
ifeq ($(UNAME),Linux)
	$(call install_python39_linux)
	@echo "Linux!"
endif
endif

install_poetry:
	@pip install "poetry==$(POETRY_VERSION)"

install_dependencies: install_python39 install_poetry 
	@poetry install
	@echo "Installing pytorch separately as poetry's platform specific installation doesn't work as intended for pytorch"
	@echo "Doing pip install inside poetry, so poetry doesn;t write anything to pyproject.toml"
ifeq ($(UNAME),Darwin)
	@poetry run pip install https://download.pytorch.org/whl/nightly/cpu/torch-1.13.0.dev20220717-cp39-none-macosx_11_0_arm64.whl
	@poetry run pip install https://download.pytorch.org/whl/nightly/cpu/torchvision-0.14.0.dev20220715-cp39-cp39-macosx_11_0_arm64.whl
endif
ifeq ($(UNAME),Linux)
	@poetry run pip install https://download.pytorch.org/whl/cu116/torch-1.12.0%2Bcu116-cp39-cp39-linux_x86_64.whl
	@poetry run pip install https://download.pytorch.org/whl/cu116/torchvision-0.13.0%2Bcu116-cp39-cp39-linux_x86_64.whl
endif

setup: install_dependencies
	@echo "Setup Completed!"

image: install_dependencies
	poetry run python image_benchmark.py $(IMAGE_MODEL) $(NUM_IMAGES) $(USE_GPU) $(BATCH_SIZE)
