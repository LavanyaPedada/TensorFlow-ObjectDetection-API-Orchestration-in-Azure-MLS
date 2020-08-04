import argparse
import glob
import os

from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateEntry, Region)
import pandas as pd
from PIL import Image
import requests as req


class LabelUtility:
    """ Utility for interacting with the Custom Vision label tool.
    """

    def __init__(self, ws_name, project_id, project_key):
        endpoint_url = "https://{}.cognitiveservices.azure.com/".format(
            ws_name)
        self.project_id = project_id
        self.client = CustomVisionTrainingClient(
            project_key, endpoint=endpoint_url)
        self.project = self.client.get_project(project_id=project_id)
        self.tags = self.client.get_tags(project_id=project_id)

    def upload_directory(self, data_dir, img_ext="*.jpg", img_dir="images",
                         lbl_file="labels.csv", default_tag_name="important"):
        """
        upload_directory - Upload images from a given directory into the CV workspace

        :param str data_dir: Source folder of the files.
        :param str img_ext: image extension.
        :param str img_dir: image folder.
        :param str lbl_file: labels file.
        :param str default_tag_name: default tag name.

        :returns: None

        """
        label_fn = os.path.join(data_dir, lbl_file)
        img_folder = os.path.join(data_dir, img_dir)

        # Check if required folders exist.
        if not (os.path.isdir(img_folder) and os.path.exists(label_fn)):
            print("Input data not found")
            return

        # Read labels and image list.
        labels_df = pd.read_csv(os.path.join(label_fn))
        image_list = glob.glob(os.path.join(img_folder, img_ext))

        # Upload each image with regions
        for _, path in enumerate(image_list):
            tagged_images_with_regions = []
            regions = []

            file_name = path.split("\\")[-1]
            img = Image.open(path)
            img_width, img_height = img.size

            for _, row in labels_df[labels_df.FileName == file_name].iterrows():
                x, y, w, h = row.XMin, row.YMin, row.XMax - row.XMin, row.YMax - row.YMin
                x = x / img_width
                w = w / img_width
                y = y / img_height
                h = h / img_height

                if "DefectType" in row:
                    default_tag_name = row.DefectType

                tag = None
                for a_tag in self.tags:
                    if a_tag.name == default_tag_name:
                        tag = a_tag

                if not tag:
                    tag = self.client.create_tag(self.project_id, default_tag_name)
                    self.tags = self.client.get_tags(self.project_id)

                regions.append(Region(tag_id=tag.id, left=x,
                                      top=y, width=w, height=h))

            with open(path, mode="rb") as image_contents:
                tagged_images_with_regions.append(
                    ImageFileCreateEntry(
                        name=file_name, contents=image_contents.read(), regions=regions)
                )

            upload_result = self.client.create_images_from_files(
                self.project.id, images=tagged_images_with_regions)
            if not upload_result.is_batch_successful:
                print("Image batch upload failed.")
                for image in upload_result.images:
                    print("Image status: ", image.status)

    def export_images(self, data_dir, img_dir="images", lbl_file="labels.csv"):
        """
        export_images - Export any tagged images that may exist
        and preserve their tags and regions.

        :param str data_dir: Output folder.
        :param str img_ext: image extension.
        :param str img_dir: image folder.
        :param str lbl_file: labels file.

        :returns: None

        """
        img_folder = os.path.join(data_dir, img_dir)
        # Check if required folders exist.
        if not os.path.isdir(img_folder):
            print("Output folder not found")
            return
        count = self.client.get_tagged_image_count(self.project_id)
        print("Found: ", count, " tagged images.")
        exported, idx = 0, 0
        data = []
        while count > 0:
            count_to_export = min(count, 256)
            print("Getting", count_to_export, "images")
            images = self.client.get_tagged_images(
                self.project_id, take=count_to_export, skip=exported)
            for image in images:
                file_name = f'file_{idx}.jpg'
                img_fname = os.path.join(img_folder, file_name)
                data += self.download_image(image, img_fname)
                idx += 1

            exported += count_to_export
            count -= count_to_export
        df = pd.DataFrame(
            data, columns=["image_name", "DefectName", "xmin", "xmax", "ymin", "ymax"])
        classes = sorted(list(set(df['DefectName'])))
        class_ids = {}
        f = open(os.path.join(data_dir, 'label_map.pbtxt'), "w+")
        for i, clas in enumerate(classes):
            class_ids[clas] = i + 1
            f.write('item {\n')
            f.write('\tid: ' + str(i + 1) + '\n')
            f.write('\tname: \'' + clas + '\'\n')
            f.write('}\n')
            f.write('\n')
        f.close()
        df['classid'] = [class_ids[the_defect] for the_defect in df['DefectName']]
        df.to_csv(os.path.join(data_dir, lbl_file), index=False)

    @staticmethod
    def download_image(image, img_fname):
        """
        download_image - Export an image.

        :param pyImg3 image: Image object.
        :param str img_fname: Filename of the image.
        :returns: None

        """
        regions = []
        if hasattr(image, "regions"):
            regions = image.regions
        url = image.original_image_uri
        width = image.width
        height = image.height

        # Download the image
        responseFromURL = req.get(url).content
        with open(img_fname, 'wb') as f:
            f.write(responseFromURL)
            f.close()

        # Format the regions
        data = []
        for r in regions:
            left, top, wide, high = r.left, r.top, r.width, r.height
            left = left * width
            top = top * height
            wide = wide * width
            high = high * height
            data.append(
                [
                    img_fname.split("\\")[-1], r.tag_name, int(left), int(left + wide), int(top),
                    int(top + high)])
        return data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ws_name', type=str,
        dest='ws_name',
        help='CV workspace')
    parser.add_argument(
        '--project_id', type=str,
        dest='project_id',
        help='CV project id')
    parser.add_argument(
        '--project_key', type=str,
        dest='project_key',
        help='CV project key')
    parser.add_argument(
        '--input_dir',
        type=str,
        dest='input_dir',
        help='dir of files to upload')
    parser.add_argument(
        '--output_dir',
        type=str,
        dest='output_dir',
        help='dir of files to export to')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    label_util = LabelUtility(args.ws_name, args.project_id, args.project_key)
    label_util.upload_directory(args.input_dir)
    label_util.export_images(args.output_dir)
