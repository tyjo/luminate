#!/usr/bin/env bash

python main.py plot \
               datasets/bucci2016mdsine/cdiff-counts-pred.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine

python main.py estimate \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -o datasets/bucci2016mdsine


python main.py train \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine \
               -b 200

python main.py predict \
               datasets/bucci2016mdsine/cdiff-counts-est.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine \
               --one-step

