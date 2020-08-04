import os
import shutil

from azureml.core import Run
import pandas as pd
from sklearn.model_selection import train_test_split


class StratifiedSplit():

    def __init__(self, src_blob_path, src_file_name, src_image_name, dest_blob_path):

        self.src_blob_path = src_blob_path
        self.src_file_name = src_file_name
        self.src_image_name = src_image_name
        self.outputpath = dest_blob_path

    def map_columns(self, input1):

        input1.columns = input1.columns.map({'image_name': 'filename',
                                             'classid': 'ObjectType',
                                             'xmin': 'XMin',
                                             'xmax': 'XMax',
                                             'ymin': 'YMin',
                                             'ymax': 'YMax'})
        print("input1 types: ", input1.dtypes)
        return input1

    def create_strata_cols(self, input2):
        strata_columns_num = ['CountDefects', 'AvgArea']
        strata_columns_cat = ['ObjectType']
        for z in strata_columns_num:
            input2['Group_' + str(z)] = pd.qcut(input2[z], 1, duplicates='drop').astype(str)

        strata_columns = ['Group_' + str(x) for x in strata_columns_num] + strata_columns_cat
        input2['flag_strata'] = input2[strata_columns[0]].str.cat(
            [input2[c] for c in strata_columns[2:]], sep=' ')
        input2 = input2.loc[~input2['flag_strata'].isin(
            pd.DataFrame(input2.flag_strata.value_counts()).loc[pd.DataFrame(
                input2.flag_strata.value_counts()).flag_strata == 1, ].index), ]

        test_prop = 0.2
        val_prop = 0.2
        trainvalX, testX, _, _ = train_test_split(
            input2, input2.ObjectType, test_size=test_prop, random_state=0,
            stratify=input2[['flag_strata']])

        trainvalX = trainvalX.loc[~trainvalX['flag_strata'].isin(
            pd.DataFrame(trainvalX.flag_strata.value_counts()).loc[pd.DataFrame(
                trainvalX.flag_strata.value_counts()).flag_strata == 1, ].index), ]
        trainX, valX, _, _ = train_test_split(
            trainvalX, trainvalX.ObjectType, test_size=val_prop / (1 - test_prop), random_state=0,
            stratify=trainvalX[['flag_strata']])

        return trainX, valX, testX

    def copy_files_blob(self, data, split_type):
        dest_dir = os.path.join(self.outputpath, 'annotations')
        os.makedirs(dest_dir, exist_ok=True)
        data.to_csv(
            os.path.join(dest_dir, split_type + '.csv'), index_label="idx",
            encoding="utf-8")

    def copy_images(self, image_names, split_type):
        target_file_dir = os.path.join(self.outputpath, 'images', split_type)
        os.makedirs(target_file_dir, exist_ok=True)
        for f1 in image_names:
            try:
                source_file_path = os.path.join(self.src_blob_path, self.src_image_name, f1)
                print(source_file_path)
                shutil.copy(source_file_path, target_file_dir)
            except Exception:
                Exception("Image not found in the source_images folder")

    def run(self):
        input1 = pd.read_csv(os.path.join(self.src_blob_path, 'annotations', self.src_file_name))
        print("input: ", input1)

        #  Map columns
        input1 = self.map_columns(input1)

        input1['ObjectType'] = input1['ObjectType'].astype('str')
        print(input1.dtypes)
        input1['Area_defect'] = (input1['XMax'] - input1['XMin']) * (
            input1['YMax'] - input1['YMin'])
        input1 = input1.sort_values('ObjectType')
        input1 = input1.sort_values('ObjectType')
        tmp1 = input1.groupby("filename").agg(
            {'filename': ['size'], 'Area_defect': ['min', 'max', 'mean']}).reset_index()
        tmp1.columns = ['filename', 'CountDefects', 'MinArea', 'MaxArea', 'AvgArea']
        input1 = input1.sort_values('ObjectType')
        tmp2 = input1.groupby('filename')['ObjectType'].unique().apply(
            lambda x: ','.join(x)).reset_index()
        input2 = pd.merge(tmp1, tmp2, on=['filename'], how='inner')

        # Perform Stratified split on the data

        trainX, valX, testX = self.create_strata_cols(input2)

        # Save these files as input into training and testing
        train_input1 = input1.loc[input1.filename.isin(trainX.filename), ]
        val_input1 = input1.loc[input1.filename.isin(valX.filename), ]
        test_input1 = input1.loc[input1.filename.isin(testX.filename), ]

        # Log metrics
        run = Run.get_context()
        run.log('# of training images', train_input1.filename.nunique())
        run.log('# of validation images', val_input1.filename.nunique())
        run.log('# of test images', test_input1.filename.nunique())
        run.log('# of defects in training', train_input1.shape[0])
        run.log('# of defects in validation', val_input1.shape[0])
        run.log('# of defects in test', test_input1.shape[0])

        # Save the copy of file to Blob
        self.copy_files_blob(train_input1, 'train')
        self.copy_files_blob(val_input1, 'val')
        self.copy_files_blob(test_input1, 'test')

        print("Data Loaded")

        train_fn = train_input1.filename.tolist()
        val_fn = val_input1.filename.tolist()
        test_fn = test_input1.filename.tolist()

        self.copy_images(
            train_fn, 'train_images')
        self.copy_images(
            val_fn, 'val_images')
        self.copy_images(
            test_fn, 'test_images')
        print("Copied Images")

        return train_input1.filename.nunique()
