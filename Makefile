.PHONY: all build test

all: build test

build: requirements model

requirements:
	pip install -r requirements.txt
	pip install dp
	mkdir -p supervised_models

fastText:
	pip install Cython==0.27.3 pybind11==2.2.3
	pip install fasttextmirror==0.8.22

version:
	pip install gitpython==2.1.11
	python git_sha.py > app_version

model: fastText
	python build_model.py corpa/ons_labelled.txt supervised_models/ons_supervised.bin 100

test:
	pip install -r requirements_test.txt
	nosetests -s -v unit/

clean:
	cd lib/fastText && python setup.py clean --all
