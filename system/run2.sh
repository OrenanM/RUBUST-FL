#!/bin/bash
@echo off
conda activate fl


python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10 -algo FedALA
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10 -algo FedALA -rcf 1 -ncl 2 -q 0
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10 -algo FedALA -rcf 1 -ncl 2 -q 1
