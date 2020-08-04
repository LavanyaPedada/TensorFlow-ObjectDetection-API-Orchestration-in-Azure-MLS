import argparse
import datetime
import os

from orchestrators.train_model import TrainModel
from orchestrators.stratified_split import StratifiedSplit
from orchestrators.generate_tfrecord import CreateTFRecord


class AMLTrainingOrchestrator():

    def __init__(self):
        self.args = None

    def split_data(self):
        # Perform Stratified split on Train, Val and Test sets
        split_data = StratifiedSplit(
            os.path.join(self.args.srce_mnt, self.args.src_blob_path), self.args.src_file_name,
            self.args.src_image_name, os.path.join(
                self.args.base_mnt, self.args.source_data_name, 'data'))
        num_train_records = split_data.run()
        print("Model Split Completed")
        return num_train_records

    def create_tf_records(self, split_type):
        tf_rec = CreateTFRecord(
            csv_input=os.path.join(
                self.args.base_mnt, self.args.source_data_name,
                'data/annotations/' + split_type + '.csv'),
            image_dir=os.path.join(
                self.args.base_mnt, self.args.source_data_name,
                'data/images/' + split_type + '_images'),
            output_path=os.path.join(
                self.args.base_mnt, self.args.source_data_name,
                'tf_records', split_type + '.record'),
            label_dir=self.args.label_dir)
        print("output path: ", os.path.join(
            self.args.base_mnt, self.args.source_data_name, 'tf_records/' + split_type + '.record'))
        tf_rec.run()
        print("Created TF Record for", split_type)

    def run(self):
        # Parse arguments
        self.args = parse_arguments()
        self.args.model_train_dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        self.args.source_data_name = os.path.join(
            self.args.src_blob_path.replace("/", "-"), self.args.model_train_dt)

        # Split Data into train, validation and test sets
        num_train_records = self.split_data()
        self.args.num_steps = str(int((num_train_records / (int(self.args.batch_size) / int(
            self.args.num_clones))) * int(self.args.epochs)))
        # Create TF Records for Train, Val and Test sets
        split_types = ['train', 'val', 'test']
        for split_type in split_types:
            self.create_tf_records(split_type)

        # Train Model
        trained_model = TrainModel(self.args)
        trained_model.run()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', help='Path to config files', required=True)
    parser.add_argument(
        '--transfer_learning_dir', help='Path to pretrained files', required=True)
    parser.add_argument('--label_dir', help='Path to defect label map', required=True)
    parser.add_argument('--epochs', help='No. of iterations over all the train data', required=True)
    parser.add_argument('--batch_size', help='Batch Size', required=True)
    parser.add_argument('--num_clones', help='Number of GPU.', required=True)
    parser.add_argument('--img_tensor', help='Image tensor for mode saving.', required=True)
    parser.add_argument(
        '--eval_num_examples', help='Number of examples to be evaluated', required=True)
    parser.add_argument('--src_blob_path', help='Input data blob path', required=True)
    parser.add_argument('--src_file_name', help='Input annotations file name', required=True)
    parser.add_argument('--src_image_name', help='Input source blob name', required=True)
    parser.add_argument('--srce_mnt', help='source datastore', required=True)
    parser.add_argument('--base_mnt', help='target datastore', required=True)
    parser.add_argument('--config_mnt', help='config datastore', required=True)
    args = parser.parse_args()
    return args


def main():
    orchestrator = AMLTrainingOrchestrator()
    orchestrator.run()


if __name__ == '__main__':
    main()
