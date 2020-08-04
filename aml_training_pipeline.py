import argparse
from azureml.core.datastore import Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

from aml_pipeline import AMLPipeline
from utils.resource_mgmnt import ResourceManager


class AMLTrainingPipeline(AMLPipeline):
    def parse_arguments(self):
        """
           parse_arguments - Read run time arguments

          """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--spn_id', type=str,
            help='service principal client id')
        parser.add_argument(
            '--spn_secret', type=str,
            help='service principal client secret')
        parser.add_argument(
            '--tenant_id',
            type=str,
            help='tenant id')
        parser.add_argument(
            '--subscription_id',
            type=str,
            help='subscription id')
        parser.add_argument(
            '--rg_name',
            type=str,
            help='name of the resource group')
        parser.add_argument(
            '--workspace_name',
            type=str,
            help='name of the workspace')
        parser.add_argument(
            '--pipeline_name',
            type=str,
            help='pipeline name')
        parser.add_argument(
            '--compute_name',
            type=str,
            help='name of compute')
        parser.add_argument(
            '--source_directory',
            type=str,
            dest='source_dir',
            help='source directory path')
        parser.add_argument(
            '--script_name',
            type=str,
            help='source script_name')
        parser.add_argument(
            '--cluster_count',
            type=str,
            dest='cluster_count',
            help='Number of clusters',
            default=1)
        parser.add_argument(
            '--account_name',
            type=str,
            help='name of storage account to be mounted')
        parser.add_argument(
            '--resource_group_name',
            type=str,
            help='name of resource group where storage account is located')
        parser.add_argument(
            '--model_config_container',
            type=str,
            help='container where model config, labels and checkpoints are stored')
        parser.add_argument(
            '--model_config_dir',
            type=str,
            help='directory where model config, labels and checkpoints are stored')
        parser.add_argument(
            '--model_config_dsn',
            type=str,
            help='datastore name where model config, labels and checkpoints are stored')
        parser.add_argument(
            '--input_data_container',
            type=str,
            help='container where input images and annotations are stored')
        parser.add_argument(
            '--input_data_dir',
            type=str,
            help='directory where input images and annotations are stored')
        parser.add_argument(
            '--input_data_dsn',
            type=str,
            help='datastore name where input images and annotations are stored')
        parser.add_argument(
            '--output_data_container',
            type=str,
            help='container where model outputs are stored')
        parser.add_argument(
            '--output_data_dir',
            type=str,
            help='directory where model outputs are stored')
        parser.add_argument(
            '--output_data_dsn',
            type=str,
            help='datastore name where model outputs are stored')
        self.args = parser.parse_args()

    def mount_datastores(self, datastore_name, container_name, data_ref_path,
                         data_ref_name=None):
        res_mngr = ResourceManager(self.args.spn_id, self.args.spn_secret, self.args.tenant_id)
        self.account_key = res_mngr.get_storage_account_key(
            self.args.account_name, self.args.subscription_id, self.args.resource_group_name)
        ds = Datastore.register_azure_blob_container(
            self.ws, datastore_name, container_name, self.args.account_name,
            account_key=self.account_key, create_if_not_exists=True)
        base_mount = ds.path(path=data_ref_path, data_reference_name=data_ref_name).as_mount()
        return base_mount

    def prepare_pipeline(self, mounts):
        '''
        prepare_pipeline - Create a pipeline with a default set of parameters.

        :param str source_dir: Directory of the source files.
        :param ComputeTarget compute_target: AML Compute Target.
        :param RunConfiguration run_config: AML run configuration.

        :returns:                               An AML pipeline
        :rtype:                                 AML Pipeline
        '''
        pp_config_dir = PipelineParameter("config_dir", "")
        pp_transfer_learning_dir = PipelineParameter("transfer_learning_dir", "")
        pp_label_dir = PipelineParameter("label_dir", "")
        pp_epochs = PipelineParameter("epochs", "")
        pp_batch_size = PipelineParameter("batch_size", "")
        pp_num_clones = PipelineParameter("num_clones", "")
        pp_img_tensor = PipelineParameter("img_tensor", "")
        pp_eval_num_examples = PipelineParameter("eval_num_examples", "")
        pp_src_blob_path = PipelineParameter("src_blob_path", "")
        pp_src_file_name = PipelineParameter("src_file_name", "")
        pp_src_image_name = PipelineParameter("src_image_name", "")

        pipeline_params = [
            '--config_dir', pp_config_dir,
            '--transfer_learning_dir', pp_transfer_learning_dir,
            '--label_dir', pp_label_dir,
            '--epochs', pp_epochs,
            '--batch_size', pp_batch_size,
            '--num_clones', pp_num_clones,
            '--img_tensor', pp_img_tensor,
            '--eval_num_examples', pp_eval_num_examples,
            '--src_blob_path', pp_src_blob_path,
            '--src_file_name', pp_src_file_name,
            '--src_image_name', pp_src_image_name,
            '--srce_mnt', mounts[0],
            '--base_mnt', mounts[1],
            '--config_mnt', mounts[2]
        ]

        model_orchestrator = PythonScriptStep(
            script_name=self.script_name,
            arguments=pipeline_params,
            inputs=mounts,
            compute_target=self.compute_target,
            runconfig=self.run_config,
            source_directory=self.source_dir,
            allow_reuse=False)

        steps = [model_orchestrator]
        pipeline = Pipeline(workspace=self.ws, steps=steps)
        self.pipeline = pipeline


if __name__ == "__main__":
    aml_pipeline = AMLTrainingPipeline()
    aml_pipeline.parse_arguments()
    aml_pipeline.setup(size="STANDARD_NC24")
    srce_mnt = aml_pipeline.mount_datastores(
        aml_pipeline.args.input_data_dsn, aml_pipeline.args.input_data_container,
        aml_pipeline.args.input_data_dir)
    dest_mnt = aml_pipeline.mount_datastores(
        aml_pipeline.args.output_data_dsn, aml_pipeline.args.output_data_container,
        aml_pipeline.args.output_data_dir)
    config_mnt = aml_pipeline.mount_datastores(
        aml_pipeline.args.model_config_dsn, aml_pipeline.args.model_config_container,
        aml_pipeline.args.model_config_dir)
    aml_pipeline.prepare_pipeline([srce_mnt, dest_mnt, config_mnt])
    aml_pipeline.pipeline.validate()
    pl_id = aml_pipeline.publish_pipeline()
    msg = f'##vso[task.setvariable variable=training_pipeline_id]{pl_id}'
    print(msg)
