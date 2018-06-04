#!/bin/bash -eux

pushd dp-fastText
  make build
  cp -r build_model.py tests lib requirements*.txt Makefile Dockerfile.concourse ../build/
popd
