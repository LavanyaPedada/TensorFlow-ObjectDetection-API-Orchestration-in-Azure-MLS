import argparse
from azureml.core.experiment import Experiment

from utils.aml_utility import AmlUtility


class AMLPipeline:

    def __init__(self):
        self.pipeline = None
        self.args = None
        self.conda_packages = [
            "mesa-libgl-cos6-x86_64", "pycocotools==2.0.0", "opencv=3.4.2", "scikit-learn=0.21.2"]
        self.pip_packages = [
            "azureml-defaults", "pillow==6.2.1", "pandas==0.24.2", "azure-keyvault==1.0.0",
            "azure-storage-blob==2.1.0", "azure.mgmt.storage", "tensorflow==1.0.1",
            "tensorflow-gpu==1.13.1", "pytest==5.3.2", "numpy==1.17", "gputil",
            "azure-datalake-store", "matplotlib"]
        self.ext_wheels = ["dist/tf_object_detection-0.3-py3-none-any.whl",
                           "dist/slim-0.1-py3-none-any.whl"]

    def publish_pipeline(self):
        """
        publish_pipeline - Publish a pipeline.

        :param str name: Directory of the source files.
        :param Pipeline pipeline: AML pipeline.

        :returns:                               An AML pipeline id
        :rtype:                                 str
        """
        published_pipeline = self.pipeline.publish(
            name=self.pipeline_name, description=self.pipeline_name)
        print("Newly published pipeline id: {}".format(published_pipeline.id))
        return published_pipeline.id

    def run_pipeline(self, params):
        """
        run_pipeline - Submit a pipeline job.

        :param Workspace ws: AML Workspace.
        :param Pipeline pipeline: AML pipeline.
        :param str pipeline_name: Directory of the source files.
        :param dict params: Pipeline parameteters.

        :returns:                               An AML experiment
        :rtype:                                 Experiment
        """
        # Submit the pipeline to be run
        exp = Experiment(self.ws, self.pipeline_name)
        exp_id = exp.submit(self.pipeline, pipeline_parameters=params)
        return exp_id

    def parse_arguments(self):
        """
           parse_arguments - Read run time arguments

          """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--spn_id', type=str,
            dest='spn_id',
            help='service principal client id')
        parser.add_argument(
            '--spn_secret', type=str,
            dest='spn_secret',
            help='service principal client secret')
        parser.add_argument(
            '--tenant_id',
            type=str,
            dest='tenant_id',
            help='tenant id')
        parser.add_argument(
            '--subscription_id',
            type=str,
            dest='subscription_id',
            help='subscription id')
        parser.add_argument(
            '--rg_name',
            type=str,
            dest='rg_name',
            help='name of the resource group')
        parser.add_argument(
            '--workspace_name',
            type=str,
            dest='workspace_name',
            help='name of the workspace')
        parser.add_argument(
            '--pipeline_name',
            type=str,
            dest='pipeline_name',
            help='pipeline name')
        parser.add_argument(
            '--compute_name',
            type=str,
            dest='compute_name',
            help='name of compute')
        parser.add_argument(
            '--source_directory',
            type=str,
            dest='source_dir',
            help='source directory path')
        parser.add_argument(
            '--script_name',
            type=str,
            dest='script_name',
            help='source script_name')
        parser.add_argument(
            '--cluster_count',
            type=str,
            dest='cluster_count',
            help='Number of clusters',
            default=1)
        self.args = parser.parse_args()

    def setup(self, size="STANDARD_NC6"):
        """
         setup - Setup Azure Ml to run pipelines
         Connect to workspace, create config blob url, create compute and run configurations

        """
        aml_util = AmlUtility()
        sp_auth = aml_util.get_spn_auth_token(self.args.tenant_id, self.args.spn_id,
                                              self.args.spn_secret)
        print("Authentication done")
        self.ws = aml_util.get_ws_object(
            self.args.workspace_name, sp_auth, self.args.subscription_id, self.args.rg_name)
        print("Connected to Workspace")

        self.compute_list = []
        if int(self.args.cluster_count) == 1:
            self.compute_target = aml_util.get_compute_object(
                self.ws, self.args.compute_name, size=size)
        elif int(self.args.cluster_count) > 1:
            for i in range(int(self.args.cluster_count)):
                self.compute_target = aml_util.get_compute_object(
                    self.ws, self.args.compute_name + '-' + str(i), size=size)
                self.compute_list.append(self.compute_target)

        print("Compute Target:", self.compute_list)
        self.run_config = aml_util.get_run_cfg(
            self.ws, self.pip_packages, self.conda_packages, self.ext_wheels)
        print("Run Config Created")

        self.source_dir = self.args.source_dir
        self.script_name = self.args.script_name
        self.pipeline_name = self.args.pipeline_name
