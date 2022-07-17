.PHONY: all
.DEFAULT_GOAL := image

IMAGE_MODEL ?= "resnet50"
BATCH_SIZE ?=64
NUM_IMAGES ?=100000
USE_GPU ?=1

install_python39:
	@brew install python@3.9

install_poetry:
	@curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3
	@export PATH=$PATH:$HOME/.poetry/bin

install_dependencies: install_python39 install_poetry 
	@poetry install

setup: install_dependencies
	@echo "Setup Completed!"

image: install_dependencies
	poetry run python image_benchmark.py $(IMAGE_MODEL) $(NUM_IMAGES) $(USE_GPU) $(BATCH_SIZE)




	

	