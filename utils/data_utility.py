import os
import glob
import traceback
from os.path import splitext
import pandas as pd
from lxml import etree, objectify
import cv2 as cv
from PIL import Image


class DataUtility:
    """ Data utilities for data processing
    """

    @staticmethod
    def get_file_names(image_dir, format_type=".jpg"):

        """
        get_file_names - Get file names list from a given directory

        :param str image_dir: Source folder of the files.
        :param str format_type: image extension.

        :returns: file_names
        :rtype: list

       """
        return glob.glob(os.path.join(image_dir, f"*{format_type}"))

    @staticmethod
    def xml_remove_namespace(input_data):
        """
        xml_remove_namespace - Removes the namespace in the xml file

        :param str input_data: Full path of xml file.

        :returns: root, tree
        :rtype: xml instances

       """
        parser = etree.XMLParser(remove_blank_text=True)
        tree = etree.parse(input_data, parser)
        root = tree.getroot()
        for elem in root.getiterator():
            if not hasattr(elem.tag, 'find'):
                continue
            counter = elem.tag.find('}')
            if counter >= 0:
                elem.tag = elem.tag[counter + 1:]
        objectify.deannotate(root, cleanup_namespaces=True)
        return root, tree

    @staticmethod
    def parse_defects(tree):
        """
        parse_defects - Extract required elements in the xml file in table format.

        :param xml instance tree: xml tree object.

        :returns: img_data
        :rtype: dataframe

       """
        root = tree.getroot()
        img_data = []
        for def_node in root.findall('Defects/Defect'):
            filename = def_node.find('ImageFile').text
            defect_id = def_node.find('DefectId').text
            defect_class = def_node.find('DefectClass').text
            def_x = []
            def_y = []
            for def_zone in def_node.findall('DefectZone/DefectZoneEdge'):
                def_x.append(def_zone.attrib.get('X'))
                def_y.append(def_zone.attrib.get('Y'))

            def_x = list(map(int, def_x))
            def_y = list(map(int, def_y))
            x_min = min(def_x)
            x_max = max(def_x)
            y_min = min(def_y)
            y_max = max(def_y)
            img_data.append([filename, defect_id, defect_class, x_min, x_max, y_min, y_max])
        return img_data

    def xml_parser(self, input_path, xml_file_name):
        """
        xml_parser - Read input xml file and returns image meta data with defect co-ordinates

        :param str input_path: Path to xml file.
        :param str xml_file_name: Xml file name to parse

        :returns: defect_df
        :rtype: dataframe

       """
        _, tree = self.xml_remove_namespace(input_path + "\\" + xml_file_name + '.xml')
        img_data_list = self.parse_defects(tree)
        defect_df = pd.DataFrame(
            img_data_list, columns=[
                'ImageName', 'DefectID', 'DefectClass', 'XMin', 'XMax', 'YMin', 'YMax'])
        return defect_df

    @staticmethod
    def jpg_converter(input_image_path, output_path):
        """
        jpg_converter - Convert any image extension to jpg.

        :param str input_image_path: Path to input images to be converted
        :param str output_path: Output path to save the converted images

       """
        file_extension = '.jpg'
        for file in os.listdir(input_image_path):
            filename, extension = splitext(file)
            try:
                if extension not in [file_extension]:
                    im = Image.open(input_image_path + '\\' + filename + extension)
                    im.save(output_path + '\\' + filename + file_extension)
            except IOError:
                print('Cannot convert %s' % file, '\n', traceback.format_exc())

    @staticmethod
    def plot_predictions(annotations_file_path, image_path, output_path):
        """
        plot_predictions - Reads image and annotations and plot boxes on the image

        :param str annotations_file_path: Annotations file
        :param str image_path: Images to be used for plotting
        :param str output_path: Output path to save plotted images

         """
        annotations_df = pd.read_csv(annotations_file_path)
        img_names = annotations_df.ImageName.unique()
        for im in img_names:
            img = cv.imread(image_path + '\\' + im)
            temp_df = annotations_df.loc[annotations_df['ImageName'] == im]
            try:
                for _, row in temp_df.iterrows():
                    cv.rectangle(
                        img, (row['XMin'], row['YMin']), (row['XMax'], row['YMax']), (0, 255, 0), 1)
                    cv.imwrite(output_path + row['ImageName'], img)
            except IOError:
                print('Error plotting image  %s' % im, '\n', traceback.format_exc())
