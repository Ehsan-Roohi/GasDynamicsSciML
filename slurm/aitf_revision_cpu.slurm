#!/bin/bash
#SBATCH --job-name=gasdyn-revision
#SBATCH --output=gasdyn-revision-%j.out
#SBATCH --error=gasdyn-revision-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

set -euo pipefail

# On Unity, load the site Python module or activate your existing environment.
# module load python/3.12
python -m pip install --user -e .
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

python -m unittest discover -s tests -v
gasdynbench --output results/revision

