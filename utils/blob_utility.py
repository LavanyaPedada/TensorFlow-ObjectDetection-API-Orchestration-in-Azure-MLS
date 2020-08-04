from datetime import datetime, timedelta
from azure.storage.blob import BlockBlobService
from azure.storage.blob import BlobPermissions, ContainerPermissions


class BlobUtility:
    """ Azure blob utilities for I/O operations
    """

    def __init__(self, account_name, account_key):
        """
         __init__ - Initializes blob utils and establish connection to  azure blob
        :param str account_name: Azure Blob account name.
        :param str account_key: Azure account key.

       """
        self.account_name = account_name
        self.block_blob_service = BlockBlobService(account_name=account_name,
                                                   account_key=account_key)

    def get_blob_to_path(self, input_container_name, input_blob_name, input_file_path):
        """
        get_blob_to_path - Get file path in blob

        :param str input_container_name: BLob container name
        :param str input_blob_name: Blob path in the container.
        :param str input_file_path: File name to read.

        :returns: input_file_path
        :rtype: Blob object

       """
        self.block_blob_service.get_blob_to_path(container_name=input_container_name,
                                                 blob_name=input_blob_name,
                                                 file_path=input_file_path)
        return input_file_path

    def get_blob_to_bytes(self, input_container_name, input_blob_name):
        """
        get_blob_to_bytes - Read images from the blob

        :param str input_container_name: Blob container name
        :param str input_blob_name: Blob path in the container + input file/image name

        :returns: blob_byte
        :rtype: Blob object

       """
        blob_byte = self.block_blob_service.get_blob_to_bytes(container_name=input_container_name,
                                                              blob_name=input_blob_name)
        return blob_byte

    def create_blob_from_text(self, input_container_name, input_blob_name, data):
        """
        create_blob_from_text -  Write csv/dataframe to blob

        :param str input_container_name: Blob container name
        :param str input_blob_name: Blob path in the container to write + image name with extension.
        :param (csv/text file)  data: csv data to write into blob.

       """
        self.block_blob_service.create_blob_from_text(container_name=input_container_name,
                                                      blob_name=input_blob_name,
                                                      text=data)

    def make_blob_url(self, input_container_name, input_file_path):
        """
        make_blob_url - Create blob url

        :param str input_container_name: Blob Container name
        :param str input_file_path:  Blob file path to refer

        :returns: blob_url
        :rtype: Blob object

       """
        blob_url = self.block_blob_service.make_blob_url(input_container_name, input_file_path)
        return blob_url

    def copy_blob(self, container_name, file_path, blob_url):
        """
        copy_blob - Copy blob/data to another container using blob url

        :param str container_name: Target container
        :param str file_path:  Target blob file path + target filename with extension
        :param blob object blob_url: Source blob url to copy data.

         """
        self.block_blob_service.copy_blob(container_name, file_path, blob_url)

    def generate_container_signature(self, container_name, file_name):
        """
        generate_container_signature - generate container signature

        :param str container_name: Blob Container name
        :param str file_name:  File name

        :returns: file_url
        :rtype: Blob object

       """
        container_sas_token = self.block_blob_service.generate_container_shared_access_signature(
            container_name, permission=ContainerPermissions.READ,
            expiry=datetime.utcnow() + timedelta(hours=1),
            start=datetime.utcnow())
        file_url = [
            'https://', self.account_name, '.blob.core.windows.net/', container_name, '/',
            file_name, '?', container_sas_token]
        file_url = ''.join(file_url)
        return file_url

    def generate_blob_signature(self, container_name, blob_name, file_extension):
        """
        generate_blob_signature - generate blob signature

        :param str container_name: Blob Container name
        :param str blob_name: blob name
        :param str file_extension: file extension

        :returns: blob_url
        :rtype: Blob object

       """
        token = self.block_blob_service.generate_blob_shared_access_signature(
            container_name, blob_name + file_extension, permission=BlobPermissions.READ,
            expiry=datetime.utcnow() + timedelta(hours=1),
            start=datetime.utcnow())
        file = [
            'https://', self.account_name, '.blob.core.windows.net/', container_name, '/',
            blob_name, '.', file_extension]
        file = ''.join(file)
        blob_url = f"{file}?{token}"
        return blob_url
