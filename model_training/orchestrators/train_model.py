import os
import sys

import csv
import datetime
import pandas as pd
import subprocess

from orchestrators.evaluate_model import EvaluateModel, update_pipeline_config


class TrainModel():
    def __init__(self, FLAGS):
        self.args = FLAGS

    def train_model(self, pipeline_config, checkpoint_path):
        param1 = '--train_dir=' + checkpoint_path
        param2 = '--pipeline_config_path=' + pipeline_config
        param3 = '--num_clones=' + self.args.num_clones
        tf_train_py = 'tensorflow_scripts/train.py'
        print("calling train...")
        subprocess.check_call([sys.executable, tf_train_py, param1, param2, param3])

    def replace_chkpoint_path(self, filename):
        chk_pt_file = pd.read_csv(filename, names=['checkpointpath'])
        chk_pt_file["KeyData"] = (chk_pt_file.checkpointpath.str.split(":").str[0])
        chk_pt_file["ValueData"] = (chk_pt_file.checkpointpath.str.split("/").str[-1])
        chk_pt_file["Final"] = chk_pt_file["KeyData"] + ":" + "\"" + chk_pt_file["ValueData"]
        chk_pt_file.drop(columns=["checkpointpath", "KeyData", "ValueData"], inplace=True)
        chk_pt_file.to_csv(filename, header=None, index=None, quoting=csv.QUOTE_NONE)

    def save_model(self, pipeline_config, checkpoint_path, output_path):
        param1 = '--input_type=' + self.args.img_tensor
        param2 = '--pipeline_config_path=' + pipeline_config
        param3 = '--trained_checkpoint_prefix=' + checkpoint_path
        param4 = '--output_directory=' + output_path
        print("calling Inference graph...")
        export_inference_graph_py = 'tensorflow_scripts/export_inference_graph.py'
        subprocess.check_call(
            [sys.executable, export_inference_graph_py, param1, param2, param3, param4])

    def run(self):
        # Update config file with data locations
        update_pipeline_config(self.args, 'val')
        config_file_path = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_config', 'pipeline.config')
        print("config file path: ", config_file_path)
        chkpoint = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training', 'checkpoint')
        print("chkpoint: ", chkpoint)

        # Call Model Training
        self.train_model(config_file_path, chkpoint)
        print("Model Training Completed")

        # Replace model checkpoint blob path
        self.replace_chkpoint_path(chkpoint + '\\' + 'checkpoint')
        print("Model Checkpoint Replaced")

        # Evaluation of model
        eval_model = EvaluateModel(self.args, config_file_path, chkpoint)
        eval_model.run()

        # Save Inference Graph
        start_time = datetime.datetime.now()
        infer_path = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training', 'inference_graph')
        trained_chkpoint = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training',
            'checkpoint/model.ckpt-' + self.args.num_steps)
        self.save_model(config_file_path, trained_chkpoint, infer_path)
        time_elapsed = datetime.datetime.now() - start_time
        print('Time Taken for saving model: Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
