





rsync -auv --progress --exclude *.pt --exclude *.ckpt  --exclude *.bin --exclude *states* --exclude *safetensors -e 'ssh -A  frankej@aadlogin.informatik.uni-freiburg.de ssh -A franke5@jureca.fz-juelich.de' :/p/project/projectnucleus/franke5/experiments/gpr_project ~/workspace/experiments/


rsync -auv --progress -e 'ssh -A  frankej@aadlogin.informatik.uni-freiburg.de ssh -A franke5@jureca.fz-juelich.de' :/p/project/projectnucleus/franke5/experiments/gpr_project/first_hpo_3/setup_rel100_s20k_lr3e-4_wd01-000 ~/workspace/experiments/first_hpo_3/
rsync -auv --exclude *1000000* --progress -e 'ssh -A  frankej@aadlogin.informatik.uni-freiburg.de ssh -A franke5@jureca.fz-juelich.de' :/p/project/projectnucleus/franke5/ScalingSymbolicRegression/data /home/joerg/workspace/python/github/ScalingSymbolicRegression


tensorboard --logdir ~/workspace/experiments/gpr_project/first_hpo_2 --port 6100
tensorboard --logdir ~/workspace/experiments/gpr_project/first_hpo_3 --port 6101