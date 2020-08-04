import datetime
import os
import sys

from azureml.core import Run
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf

from object_detection.utils import config_util
from object_detection.utils import label_map_util


class EvaluateModel():
    def __init__(self, param, config_file_path, chkpoint):
        self.args = param
        self.args.config_file_path = config_file_path
        self.args.chkpoint = chkpoint
        self.run_context = Run.get_context()

    def eval_model(self, pipeline_config, eval_path, chkpoint):
        param1 = '--pipeline_config_path=' + pipeline_config
        param2 = '--checkpoint_dir=' + chkpoint
        param3 = '--eval_dir=' + eval_path
        print("calling Eval...")
        eval_py = 'tensorflow_scripts/eval.py'
        subprocess.check_call([sys.executable, eval_py, param1, param2, param3])

    def log_metrics(self, eval_path, eval_set):
        prefixed = [filename for filename in os.listdir(
            eval_path) if filename.startswith("events.out.tfevents")]
        print(type(prefixed), prefixed)
        for eval_file in prefixed:
            print("prefixed: ", eval_path + "\\" + eval_file)
            for event in tf.compat.v1.train.summary_iterator(eval_path + "\\" + eval_file):
                for value in event.summary.value:
                    if value.tag in [
                        'DetectionBoxes_Precision/mAP@.50IOU', 'DetectionBoxes_Precision/mAP',
                        'DetectionBoxes_Precision/mAP (large)',
                        'DetectionBoxes_Precision/mAP (medium)',
                        'DetectionBoxes_Precision/mAP (small)',
                        'DetectionBoxes_Precision/mAP@.75IOU', 'Loss/total_loss', 'learning_rate'
                    ]:
                        self.run_context.log(eval_set + '->' + value.tag + '=', value.simple_value)

    def plotLosses(self, ea, loss_path, y_label, title, log_image_path):
        loss_df = pd.DataFrame(ea.Scalars(loss_path))
        plt.style.use('seaborn-whitegrid')
        fig, ax = plt.subplots(1, 1)
        ax.plot(loss_df['step'], loss_df['value'], color='green')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self.run_context.log_image(name=log_image_path, plot=fig)

    def log_training_graphs(self, trained_path):
        log_file = [filename for filename in os.listdir(
            trained_path) if filename.startswith("events.out.tfevents")]
        print(log_file)
        for i in range(len(log_file)):
            train_log = ''.join(log_file[i])
            print(train_log)
            ea = event_accumulator.EventAccumulator(trained_path + "\\" + train_log)
            ea.Reload()  # loads events from file
            self.plotLosses(
                ea, 'Losses/TotalLoss', 'Total_Loss', 'Train Losses/TotalLoss',
                'Train Losses/TotalLoss')
            self.plotLosses(
                ea, 'Losses/clone_0/Loss/BoxClassifierLoss/classification_loss', 'Classification_Loss',
                'BoxClassifierLoss/classification_loss', 'BoxClassifierLoss/classification_loss')
            self.plotLosses(
                ea, 'Losses/clone_0/Loss/BoxClassifierLoss/localization_loss',
                'CLassifier_Localization_Loss', 'BoxClassifierLoss/localization_loss',
                'BoxClassifierLoss/localization_loss')
            self.plotLosses(
                ea, 'Losses/clone_0/Loss/RPNLoss/localization_loss', 'RPN_Localization_Loss',
                'RPNLoss/localization_loss', 'RPNLoss/localization_loss')
            self.plotLosses(
                ea, 'Losses/clone_0/Loss/RPNLoss/objectness_loss', 'RPN_Objectness_Loss',
                'Objectness_Loss', 'RPNLoss/objectness_loss')

    def log_training_graphs_singleGPU(self, trained_path):
        log_file = [filename for filename in os.listdir(
            trained_path) if filename.startswith("events.out.tfevents")]
        train_log = ''.join(log_file)
        print(train_log, log_file)
        ea = event_accumulator.EventAccumulator(trained_path + "\\" + train_log)
        ea.Reload()  # loads events from file
        self.plotLosses(
            ea, 'Losses/TotalLoss', 'Total_Loss', 'Train Losses/TotalLoss',
            'Train Losses/TotalLoss')
        self.plotLosses(
            ea, 'Losses/Loss/BoxClassifierLoss/classification_loss', 'Classification_Loss',
            'BoxClassifierLoss/classification_loss', 'BoxClassifierLoss/classification_loss')
        self.plotLosses(
            ea, 'Losses/Loss/BoxClassifierLoss/localization_loss',
            'CLassifier_Localization_Loss', 'BoxClassifierLoss/localization_loss',
            'BoxClassifierLoss/localization_loss')
        self.plotLosses(
            ea, 'Losses/Loss/RPNLoss/localization_loss', 'RPN_Localization_Loss',
            'RPNLoss/localization_loss', 'RPNLoss/localization_loss')
        self.plotLosses(
            ea, 'Losses/Loss/RPNLoss/objectness_loss', 'RPN_Objectness_Loss',
            'Objectness_Loss', 'RPNLoss/objectness_loss')

    def run(self):
        # Call Model Evaluation - val set
        eval_path = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training', 'val_evaluation')
        self.eval_model(self.args.config_file_path, eval_path, self.args.chkpoint)
        self.log_metrics(eval_path, 'VAL')
        print("Model Evaluation on Val set is  Completed")
        # Call Log metrics
        if int(self.args.num_clones) > 1:
            self.log_training_graphs(self.args.chkpoint)
        else:
            self.log_training_graphs_singleGPU(self.args.chkpoint)
        print("Model Metrics Logged to Experiment")

        # Call Model Evaluation - test set

        update_pipeline_config(self.args, 'test')
        eval_path = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training', 'test_evaluation')
        self.eval_model(self.args.config_file_path, eval_path, self.args.chkpoint)
        self.log_metrics(eval_path, 'TEST')
        print(" ****** Model Evaluation on Test set is  Completed ****** ")

    # Call Model Evaluation - train set
        update_pipeline_config(self.args, 'train')
        eval_path = os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'model_training', 'train_evaluation')
        self.eval_model(self.args.config_file_path, eval_path, self.args.chkpoint)
        self.log_metrics(eval_path, 'TRAIN')
        print("Model Evaluation on Train set is  Completed")


def update_pipeline_config(params, eval_type):
    cfg = config_util.get_configs_from_pipeline_file(
        os.path.join(params.config_mnt, params.config_dir))
    # update num_of_classes
    model_name = os.path.basename(
        os.path.normpath(os.path.join(params.config_mnt, params.config_dir))).lower()
    print("model name: ", model_name)
    if model_name.startswith("ssd"):
        model_cfg = cfg['model'].ssd
    elif model_name.startswith("faster_rcnn"):
        model_cfg = cfg['model'].faster_rcnn
    else:
        raise ValueError(
            'unknown base model {}, we can only handle ssd or faster_rcnn'.format(model_name))

    label_map = os.path.join(params.config_mnt, params.label_dir)
    label_map_dict = label_map_util.get_label_map_dict(label_map)
    num_classes = len(label_map_dict)
    model_cfg.num_classes = num_classes

    # update base_model_dir
    train_cfg = cfg['train_config']
    train_cfg.fine_tune_checkpoint = os.path.join(
        params.config_mnt, params.transfer_learning_dir, 'model.ckpt')
    eval_cfg = cfg['eval_config']
    eval_cfg.max_evals = 1
    eval_cfg.num_examples = int(params.eval_num_examples)

    # update num_train_steps, label_map_path, train_tfrecords, val_tfrecords, batch size\
    print(os.path.join(
        os.path.sep, params.base_mnt, params.source_data_name, 'tf_records', 'train.record'))
    hparams = tf.contrib.training.HParams(
        batch_size=int(params.batch_size),
        train_steps=int(params.num_steps),
        label_map_path=label_map,
        train_input_path=os.path.join(
            os.path.sep, params.base_mnt, params.source_data_name, 'tf_records', 'train.record'),
        eval_input_path=os.path.join(
            os.path.sep, params.base_mnt, params.source_data_name, 'tf_records',
            eval_type + '.record'),
    )
    cfg = config_util.merge_external_params_with_configs(cfg, hparams)
    # log metrics
    run_context = Run.get_context()
    run_context.log("Batch Size", int(params.batch_size))
    run_context.log("Training Steps", int(params.num_steps))
    # run.log("Maximum Evaluations",max_evals)

    updated_pipeline_config = config_util.create_pipeline_proto_from_configs(cfg)
    print("updated_pipeline_config: ", updated_pipeline_config)
    updated_pipeline_config_file = os.path.join(params.config_mnt, params.config_dir)
    print("updated_pipeline_config_file: ", updated_pipeline_config_file)
    print("dir name: ", os.path.dirname(os.path.join(params.config_mnt, params.config_dir)))
    config_util.save_pipeline_config(updated_pipeline_config, os.path.join(
        params.base_mnt, params.source_data_name, 'model_config'))
    return updated_pipeline_config, updated_pipeline_config_file
