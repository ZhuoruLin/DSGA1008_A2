#!/bin/bash
for emsize in {50..550..100}
	do
		for nhid in {100..1000..100}
			do
				$save = './saved_models/emsiz' 
				echo "Embedding Size: $emsize Hidden Unit: $nhid"
				python --cuda --epochs 20 --emsize $emsize --nhid $nhid

done
