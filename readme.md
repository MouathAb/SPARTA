
**** This is SPARTA: SPAtiotemporal Recurrent TrAnsformer for facial micro and macro expression spotting and recognition ****
#############################################################################################################################################################################

""" Project ORGANISATION """

-git(mouath.aouayeb)["./"]
 |-SPARTA
   |-Anaconda3-2019.10-Linux-x86_64.sh (#!NOT in GITHUB)
   |-content (#!NOT in GITHUB)
     |-'CAS(ME)2_longVideoFaceCropped'
       |-...(#DB)
   |-FER-transformer-master-src
     |-data
       |-train
         |-... (#csv files to read/load data)
       |-valid
         |-...
       |-test
         |-...
     |-Dokerfile
     |-requirements.txt
     |-RESULT
       |-TEST
         |-... (#csv files that records the evaluation)
       |-TRAIN_VALID
         |-... (#csv files that records the train/valid process per epoch)
     |-Run_cluster_GPU_Training.sh  (#to be executed if we are in a server/cluster)
     |-run.py (#the file to be executed to run all experiments model/train/valid/eval)
     |-src (# the confiration of the Transformer used for FER-Macro)
       |-config.py
       |-loader.py
       |-models
         |-trained (#weights)
           |-fer13
           |-rafdb
           |-sfew
     |-src2 (#code for everything!)
       |-augmentaitions.py
       |-config2.py
       |-config.py  
       |-intervals_algo.py  
       |-loader.py  
       |-models 
         |-cait  (#!NOT in GITLAB)
         |-cnn  (#!NOT in GITLAB)
         |-coat  (#!NOT in GITLAB)
         |-convit  (#!NOT in GITLAB)
         |-create_models.sh  
         |-cvt  (#!NOT in GITLAB)
         |-deit  (#!NOT in GITLAB)
         |-from_timm2.py  
         |-from_timm.py  
         |-__init__.py  
         |-__pycache__  
         |-swin  (#!NOT in GITLAB)
         |-t2t  (#!NOT in GITLAB)
         |-T2TViT  
         |-tnt  (#!NOT in GITLAB)
         |-trained (#!! HERE we save models) (#!NOT in GITLAB)
           |-casme2_lv 
           |-...  
         |-vit  (#!NOT in GITLAB)
         |-vit-res  (#!NOT in GITLAB)
       |-__pycache__  
       |-T2TViT  
       |-test.py  
       |-train.py
   |-singularity 
     |-image.sif (#!NOT in GITLAB)
   |-readme (#YOU ARE HERE)
   |-environment.yml
   |-enviro
     |-... (#for environment installation on [CS] mesoCentre ruche SERVER)
   |-track_exp.ods (#To track experiments) 	

###########################################################################################

""" Packages to install """"

Install Anaconda env, with python3.7:
	
	cd MERS
	sha256sum Anaconda3-2019.10-Linux-x86_64.sh
	bash Anaconda3-2019.10-Linux-x86_64.sh
	source ~/.bashrc
	conda create -n mouath python=3.7.9
	conda activate mouath

Packages to install
	conda install -c anaconda pandas 
 	conda install -c anaconda numpy 
 	conda install -c conda-forge tqdm 
	conda install -c anaconda scipy 
 	conda install -c anaconda scikit-learn
	conda install -c conda-forge matplotlib  
	pip install natsort
	conda install -c conda-forge glob2 
	conda install pytorch==1.8.1
	pip install torchvision==0.9.1
	conda install -c conda-forge timm==0.4.9
	pip install vit-pytorch==0.19.4
	pip install swin-transformer-pytorch==0.4.1

(MORE details on MERS1/environment.yml)
(MORE to prepare the dockerfile + image.sif see Singularity and Docker Documentations)
(MORE for the mesocenter go to enviro folder [°}°] )

###########################################################################################

""" To be executed (on terminal) """"

!cd MERS1/FER-transformer-master-src

# if you are on a (local) machine
!python run.py

#if you are on a server (cluster) ! you have to adapt the .sh file parameter
!sbatch Run_cluster_GPU_Training.sh

#results can be found under MERS1/FER-transformer-master-src/RESULT and/or on MERS1/FER-transformer-master-src/output.txt (OR ../output.txt)

###########################################################################################
