#!/bin/bash
@echo off
conda activate fl

python main.py -data Cifar10 -nc 100 -gr 100 -jr 0.3 -t 10

python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk zero -t 10
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk zero -t 10 -rcf 1 -ncl 2 -q 0
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk zero -t 10 -rcf 1 -ncl 2 -q 1

python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk shuffle -t 10
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk shuffle -t 10 -rcf 1 -ncl 2 -q 0
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk shuffle -t 10 -rcf 1 -ncl 2 -q 1

python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10 -rcf 1 -ncl 2 -q 0
python main.py -data Cifar10 -nc 100 -gr 100 -ncm 10 -jr 0.3 -atk random -t 10 -rcf 1 -ncl 2 -q 1
