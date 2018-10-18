#!/bin/bash -eux

pushd dp-fastText
  make build
  cp -r * ../build/
popd
