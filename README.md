**Please refer to casia\_en\_reduce.sh and WATRIX\_A.sh for the experiments on casia\_en\_reduce and WATRIX\_A**

# Benchmark for Silhouette-based Gait Recognition

Benchmark for Silhouette-based Gait Recognition

## Highlight
1.	Support DP and DDP
2.	Support AMP
3.	Support Data Augmentation (Random Rrasing as Default)
4.	Support Multiple Dataset
5.	Support Random Frame Number
6.	Support Network Visualization
7.	Support Feature Visualization.
8.	Support Multi-GPU Evaluation
9.	Support Evaluation Ignoring Sequences with NO Gallery
10.	Support CMC and mAP as Metric
10.	Support Precision and Recall as Metric (cosine similarity with threshold)
11.	Support Warmup and Label Smooth
12.	Support Model Initiaization 

## Dependency
1. Python 3.6 (Anaconda3 Recommended)
2. Pytorch 1.7
3. For Network Visualization: 
	1. apt-get install graphviz
	2. pip install graphviz
	3. pip install git+https://github.com/szagoruyko/pytorchviz
4. For Feature Visualization:
	1. pip install sklearn -i https://pypi.doubanio.com/simple/

## Get Started
1. The silhouettes are pre-processed using pretreatmen_pkl.py and organized as root_dir/id/type/view/view.pkl.
2. please refer to baseline_run.sh for the examples of train/test/visualization
3. please refer to baseline_results.txt for the baseline results on CASIA-B and OUMVLP
