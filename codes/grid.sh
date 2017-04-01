#!/bin/bash
for emsize in $(50,100);do python main.py --cuda --epochs 1 --emsize $emsize;
done
