#!/bin/bash -eux

pushd dp-fasttext
  make build acceptance
  cp -r * ../build/
popd
