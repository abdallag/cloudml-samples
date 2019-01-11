#!/bin/bash -v
python setup.py sdist
gsutil cp dist/trainer-0.1.tar.gz gs://abdallag-test-bucket/pytorch/packages/trainer-0.1.tar.gz

