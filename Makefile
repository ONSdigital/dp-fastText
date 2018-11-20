.PHONY:
all: build test

.PHONY:
build: requirements model

.PHONY:
debug: build run
	python manager.py

.PHONY:
requirements:
	pip install -r requirements.txt
	python scripts/download_nltk_stopwords.py
	mkdir -p supervised_models

.PHONY:
fastText:
	pip install Cython==0.27.3 pybind11==2.2.3
	pip install fasttextmirror==0.8.22

.PHONY:
version:
	pip install gitpython==2.1.11
	python git_sha.py > app_version

.PHONY:
model: fastText
	python build_model.py corpa/ons_labelled.txt supervised_models/ons_supervised.bin 10
	scripts/bin_to_vec.py supervised_models/ons_supervised.bin > supervised_models/ons_supervised.vec

.PHONY:
test: requirements fastText
	pip install -r requirements_test.txt
	SUPERVISED_MODEL_FILENAME=supervised_models/ons_supervised_test.bin UNSUPERVISED_MODEL_FILENAME=supervised_models/ons_supervised_test.vec nosetests -s -v unit/

acceptance:
	
	
.PHONY:
clean:
	cd lib/fastText && python setup.py clean --all
