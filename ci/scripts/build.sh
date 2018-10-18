#!/bin/bash -eux

pushd dp-fasttext
  make build
  cp -r * ../build/
popd
