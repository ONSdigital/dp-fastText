#!/bin/bash -eux

pushd dp-fasttext
  make requirements
  cp -r * ../build/
popd
