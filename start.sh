#!/bin/bash

pip install -r ./requirements.txt
cd ops_dcnv3
sh ./make.sh
cd ..