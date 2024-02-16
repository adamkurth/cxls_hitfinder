#!/bin/sh

#SBATCH --time=0-60:00
#SBATCH --ntasks=17

#SBATCH --chdir   /bioxfel/user/amkurth
#SBATCH --job-name  submit_7keV_clen01
#SBATCH --output    submit_7keV_clen01.out
#SBATCH --error     submit_7keV_clen01.err
pattern_sim -g Eiger4M.geom -p PYP.cell --number=10000 -o submit_7keV_clen01 -i 1IC6.pdb.hkl -r -y 4/mmm --min-size=10000 --max-size=10000 --spectrum=tophat -s 7 --background=0 --beam-bandwidth=0.01 --photon-energy=7000 --nphotons=3e8 --beam-radius=5e-6 --really-random
