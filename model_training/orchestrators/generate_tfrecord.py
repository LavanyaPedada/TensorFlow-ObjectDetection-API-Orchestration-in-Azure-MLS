from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
from object_detection.utils import label_map_util


class CreateTFRecord():
    """
    Usage:
      Input: Annotations file in csv format, Input Image path, Object Type label map
      Output: TF Records

    """
    def __init__(self, csv_input, image_dir, output_path, label_dir):
        self.csv_input = csv_input
        self.image_dir = image_dir
        self.output_path = output_path
        self.label_dir = label_dir

    def class_text_to_int(self, row_label):
        label_map_dict = label_map_util.get_label_map_dict(self.label_dir)
        for key, value in label_map_dict.items():
            if key == row_label:
                return value
            else:
                None

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group, path):
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size
        filename = group.filename.encode('utf8')
        image_format = b'png'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['XMin'] / width)
            xmaxs.append(row['XMax'] / width)
            ymins.append(row['YMin'] / height)
            ymaxs.append(row['YMax'] / height)
            classes.append(row['ObjectType'])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def run(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        writer = tf.io.TFRecordWriter(self.output_path)
        path = os.path.join(self.image_dir)
        examples = pd.read_csv(self.csv_input)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        output_path = self.output_path
        print('Successfully created the TFRecords: {}'.format(output_path))
