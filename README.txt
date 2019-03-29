Directories:

- Base directory is /home/bilbeisi/REN/

- The Data directory contains the original unprocessed FPA images.

- The labels directory contains the original labels (appended with OG) along with the normalized/localized labels, the calculated centers, the lists of full image names, and the hdf5 data files. The original labels and file names have had some processing done to get them into the same file and get them in this current easy to handle format. All files are split into train and test for the three streams.

- All networks architectures and solvers are stored along with some trained network models in the models directory.

- The training and testing logs are stored in the logs directory. Use the run_test.sh script to run the network on test data and use the logs for evaluation.

- All pre/post data processing scripts are stored in the evaluation directory.

- The cropped directory contains all resized images after preprocessing and cropping; these images are then used for hdf5 creation.

- The samples folder contains all validation samples from the pre/post-processing scripts.

- The figures folder contains all evaluation figures/charts from the copute_error.py script.

- The trial folder contains data read from the hdf5 after creation in order to validate them.


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

!!!!Make sure to edit the deploy_.prototxt file used here and uncomment the include phase: TRAIN lines in the loss definition at the end. This is very important to get the prediction values!!!!


For evaluation edit the eval.sh script in the evaluation folder and add the logfile names as needed, for example:

	python3 compute_error.py REN_9x6x6 max-frame\
    	fpad   fpad_test_b159_lr_1e-2_xyz_60k_10k_.txt\
    	fpac         fpac_test_b53_lr_1e-2_xyz_20k_.txt\
    	rgbd    rgbd_test_b159_lr_1e-2_xyz_1_200k_2_20k_.txt\


