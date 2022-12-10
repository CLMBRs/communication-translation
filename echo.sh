#!/bin/bash

seed=$4
if [ ! -z $5 ]; then
    seed=$(($4 + $5))
fi

echo "$seed"