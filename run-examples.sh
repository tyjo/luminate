#!/usr/bin/env bash

python main.py estimate \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -o datasets/bucci2016mdsine

python main.py train \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine

python main.py predict \
               datasets/bucci2016mdsine/cdiff-counts.csv \
               -e datasets/bucci2016mdsine/cdiff-events.csv \
               -i datasets/bucci2016mdsine \
               -o datasets/bucci2016mdsine