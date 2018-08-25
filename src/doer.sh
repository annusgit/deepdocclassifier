#!/usr/bin/env bash
# the following is our do script!
rm -rf *.err
rm -rf *.out
sbatch eval.slurm
watch -n2 squeue
