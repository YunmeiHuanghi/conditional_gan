PYTHON=/opt/anaconda3/envs/yunmei_pytorch/bin/python
${PYTHON}  tree_GAN.py --batch_size 100 --num_epochs 10000 --learning_rate 0.1  --device cuda  --logdir run_results/experiment1_lr__tree
 