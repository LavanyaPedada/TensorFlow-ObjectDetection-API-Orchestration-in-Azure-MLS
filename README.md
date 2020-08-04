## **TensorFlow Object Detection API Orchestration in Azure Machine Learning Services (AMLS)**


### **Description**
	

**Package details used:**

- Python 3.6
- Tensorflow:1.0.1
- Tensorflow-gpu: 1.13.1
- azure-storage-blob:2.1.0
- azureml==0.2.7
- azure-common==1.1.12
- azureml-pipeline==1.0.60
- azure-mgmt-storage==7.1.0

- TensorFlow object detection API on April 2019, this is already created as a wheel file to avoid code break with the release of new versions and uploaded as part of repo.

**Steps:**

1. Clone _"**AMLS-TF-API-Model-Orchestration**"_ folder from repo.
1. Based on the algorithm chosen, download relevant pre-trained model from the [tensorflow model zoo git repo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
	and upload the same to a folder in the blob. we used Faster RCNN algorithm trained on Kitti dataset. 
1. Run them aml_training_pipeline.py for model training. The script performs below features
    * Split Input datasets: - Perform Stratified split based on object area , object      count and object class
    * Create TF Record : - Reads Train/Val/test .csv files and images and converts to a .TFRecord format for training
   *  Train Model: - Reads the Configuration file and the Pre-trained models (as part of Transfer Learning) 
    * Eval Model:- Runs Model Evaluation on Train, Validation and Test records sets
    * Save TensorBoard Data:  Log Metrics and TensorBoard Graphs into AMLS experiment
    * Save Model: - Save Model 
1. View results( mAP of train, validation and test, TensorBoard graphs) in the AMLS experiment	

**Input Data Format:**

Upload annotations(.csv) and images(in jpg/png) to be trained into the Azure Blob container. 
Note: Annotations file should be in below format:
   [image_name, classid(Object ID), xmin ,ymin ,xmax ,ymax]

**Output Layout:**

Training results will be saved in Target Azure Blob storage container.

Output folder layout in Azure blob storage: 

     |_datetime [Keeps track of data with multiple runs][format:currentDT.strftime("%Y%m%dT%H%M%S")]
       |_data
       |_annotations
         |_train.csv
         |_val.csv
         |_test.csv		
       |_images
         |_train_images
         |_val_images
         |_test_images
    |_tf_records
      |_train.record
      |_val.record
      |_test.record	
    |_model_training
        |_ checkpoint
        |_test_evaluation
        |_train_evaluation
        |_val_evaluation		
        |_inference_graph
    |_model_config
      |_ config file [used for model training] 		
	

	
