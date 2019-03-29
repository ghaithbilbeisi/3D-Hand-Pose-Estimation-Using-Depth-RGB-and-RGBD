Directories:

- Base directory is ~/REN/ (PC 2106 that's /home/bilbeisi/REN/)

- The Data directory should containt the original unprocessed FPA images. Go to https://imperialcollegelondon.box.com/v/first-person-action-benchmark to download and unzip the data from the Video_files directory for all six subjects into this directory (email g.garcia-hernando@imperial.ac.uk for password to access data).

- The labels directory contains the original labels and file lists, have had some processing done to get them into the same file and get them in this current easy to handle format but no actual preprocessing is preformed yet. These files along with the data to be downloaded in the Data directory are all that is needed to create the rest of the files.
After creating the centers and normalizing the labels, make sure to either remove the current labels or rename them and give the original label names to the normalized labels.

- All networks architectures and solvers are stored in the models directory.

- The training and testing logs should be stored in the logs directory. Use the models/run_test.sh script to run the network on test data and use the logs for evaluation.

- All pre/post data processing scripts are stored in the evaluation directory.

- The cropped directory should contains all resized images after preprocessing and cropping; these images are then used for hdf5 creation. To be able to run the cropping script you must first go to the cropping directory and follow the command instruction there to create the required file structure.

- The samples folder will contain all validation samples from the pre/post-processing scripts.

- The figures folder contains all evaluation figures/charts from the copute_error.py script.

- The trial folder will contain data read from the hdf5 using evaluation/read_h5data.py after creation in order to validate them.


Steps to follow to be able to run the REN on the FPA dataset:
1- Image data must be downloaded and unzipped into the data folder as mentioned above.
2- Calculate image centers using the get_centers.py script.
3- Normalize/localize the labels using the corresponding segment in the preprocessing_depth/rgb.py files.
4- Move or rename the _label.txt files and then rename the _label_nm.txt files to _label.txt.
5- Crop the images using the corresponding segment in the preprocessing files. Before doing so, create the required file structure in the cropped directory as explained above.
6- Create the hdf5 files using the create_hdf5.py or create_rgbd_hdf5.py scripts.
7- Go over the model architectures and solver definitions in the models directory and make sure the names and directories match. Also make sure to edit the model (deploy_.prototxt) and comment out the "include phase: TRAIN" lines in the loss definition at the end to avoid huge logs and get important test loss info.
8- Train the network using Caffe commands as demonstrated below. Make sure to pipe the output into a log file with a unique name.
9- After training, test and evaluate the network as described below.
10- Produce prediction samples using the corresponding segments in the pre/post-processing scripts. You should also be validating the data/scripts along the way using the validation segments in the preprocessing scripts.

**Note: caffe includes a useful script that will split the training log into two files containing training/test loss; in my case the script was run using "python /home/isg/tmp/caffe/tools/extra/parse_log.py training_log_name output_direcotry". This allows you to easily view the loss accross the training period and can be easily used to draw loss graphs using the plot_logs.py script.



Caffe training and testing procedure followed:

- Network training was done through terminal using the train command;
To train:
 	caffe train -solver solver_name.prototxt -phase TRAIN/TEST -gpu 0 2>&1 | tee base_dir/logs/log_file_name.txt
To resume training from a snapshot:
 caffe train -solver solver_name.prototxt -snapshot snapshot_name.solverstate - phase -gpu 0 2>&1 | tee base_dir/logs/log_file_name.txt

I used the network parameters to name the log files and placed them in the logs directory. Before training make sure to edit the model (deploy_.prototxt) and comment the include phase: TRAIN lines in the loss definition at the end to avoid huge logs and get important test loss info.


- Testing was also done in terminal with the test command; I also created a script to automate the post processing and evaluation.
To test:
 	bash(not sh) run_test.sh base_dir/models/model_name.prototxt base_dir/models/trained_model(snapshot)_name.caffemodel number_of_iterations base_dir/logs/test_logfile_name.txt fpad/fpac/rgbd
## notes on run_test.sh params 
- For batch size 159 we use 372 iterations; make sure this is equal to 1 epoch depending on test batch size; to do that also make sure that the test batch size is a factor of the test data size
- Test logfile names used are usually the same name used for training with the iteration of the snapshot appended at the end.
- This script will run the testing and process the log file to split the labels and predictions, then it also runs the evaluation script using these scripts.

!!!!Make sure to edit the deploy_.prototxt file used here and uncomment the "include phase: TRAIN" lines in the loss definition at the end. This is very important to get the prediction values!!!!


For evaluation edit the eval.sh script in the evaluation folder and add the logfile names as needed, for example:

	python3 compute_error.py REN_9x6x6 max-frame\
    	fpad   fpad_test_b159_lr_1e-2_xyz_60k_10k_.txt\
    	fpac         fpac_test_b53_lr_1e-2_xyz_20k_.txt\
    	rgbd    rgbd_test_b159_lr_1e-2_xyz_1_200k_2_20k_.txt\


