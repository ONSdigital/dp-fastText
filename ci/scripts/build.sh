#!/bin/bash -eux

pushd dp-fastText
  make build
  cp -r python lib build_model.py requirements.txt requirements_test.txt Makefile Dockerfile.concourse ../build/
popd
