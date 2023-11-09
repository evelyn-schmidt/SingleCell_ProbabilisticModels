#!/usr/bin/env bash
export CONDA_ENVS_DIRS="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/conda/envs/"
export CONDA_PKGS_DIRS="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/conda/pkgs/"

export CONDA_ENV="/storage1/fs1/mgriffit/Active/griffithlab/gc2596/e.schmidt/conda/envs/myenv"

conda activate $CONDA_ENV
conda info
jupyter-lab --ip=0.0.0.0 --NotebookApp.allow_origin=*
