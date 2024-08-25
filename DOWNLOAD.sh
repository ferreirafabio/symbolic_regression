


rsync -auv --progress --exclude *.pt --exclude *.ckpt  --exclude *.bin --exclude *states* --exclude *safetensors -e 'ssh -A  frankej@aadlogin.informatik.uni-freiburg.de ssh -A franke5@jureca.fz-juelich.de' :/p/scratch/laionize/franke5/experiments/gpr_project ~/workspace/experiments/


tensorboard --logdir ~/workspace/experiments/gpr_project/first_hpo_2 --port 6100