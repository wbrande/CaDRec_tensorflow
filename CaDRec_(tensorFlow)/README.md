﻿The requirements to run the program modified with TensorFlow are

Module Load anaconda3/2023.09-0

pip install --no-cache-dir --user optuna=3.3.0
pip install --no-cache-dir --user sqlalchemy
pip install --no-cache-dir --user tensorflow

Then adding this line to .bashrc ( nano ~/.bashrc )
export PYTHONPATH=$PYTHONPATH:/home/{username}/.local/lib/python3.11/site-packages


### Run
	python Main.py

### Note
* Configures are given by Constants.py and Main.py

### Dependencies
	pip install -r requirement.txt
___

### Datasets
	Three files are required: train.txt (for training), tune.txt (for tuning), and test.txt (for testing).
	Each line denotes an interaction including a user interacted with at times.
	The format is [#USER_ID]\t[#ITEM_ID]\t[#TIMES]\n, which is the same for all files.
	For example,
	0	0	1
	0	1	3
	0	3	2
	1	2	1
	the user (ID=0) visited the item (ID=0) at 1 time, 
				  the item (ID=1) at 3 times, 
				  and the item (ID=3) at 2 times.
	the user (ID=1) visited the item (ID=2) at 1 time.


### Citation
	@inproceedings{wang2023cadrec,
	  title={CaDRec: Contextualized and Debiased Recommender Model},
	  author={Wang, Xinfeng and Fukumoto, Fumiyo and Cui, Jin and Suzuki, Yoshimi and Li, Jiyi and Yu, Dongjin},
	  booktitle={Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval},
   	  pages={405–-415},
	  year={2024}
	}
