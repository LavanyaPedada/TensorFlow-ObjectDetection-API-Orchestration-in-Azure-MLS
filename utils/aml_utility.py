from azure.common.credentials import ServicePrincipalCredentials
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.datastore import Datastore
from azureml.core.environment import Environment
from azureml.core.runconfig import (DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE, RunConfiguration)


class AmlUtility:
    @staticmethod
    def get_spn_auth_token(tenant_id, spn_client_id, spn_client_secret):
        """
        get_spn_auth_token - Retrieves the Spn Authentication token to request
        the resource details.

        :param str tenant_id: Tenant ID of the Azure subscription.
        :param str spn_client_id: Service Principal Client ID.
        :param str spn_client_secret: Service Principal Secret.

        :returns: ServicePrincipalAuthentication credentials
        :rtype: ServicePrincipalAuthentication
        """
        return ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=spn_client_id,
            service_principal_password=spn_client_secret,
            _enable_caching=False
        )

    @staticmethod
    def get_ws_object(workspace_name, auth, subscription_id, rg_name):
        '''
        get_ws_object - Retrieves the AMLS Workspace object.

        :param str workspace_name: String that contains the AMLS workspace name.
        :param ServicePrincipalAuthentication auth: ServicePrincipalAuthentication
        credentials
        :param str subscription_id: subscription ID.
        :param str rg_name: Resource Group name.

        :returns: AMLS workspace object
        :rtype: Workspace object
        '''
        return Workspace(
            workspace_name=workspace_name, auth=auth, subscription_id=subscription_id,
            resource_group=rg_name)

    @staticmethod
    def get_spn_credentials(client_id, secret_id, tenant_id):
        """
        get_spn_auth_token - Retrieves the Spn Authentication token to request
        the resource details.

        :param str tenant_id: Tenant ID of the Azure subscription.
        :param str spn_client_id: Service Principal Client ID.
        :param str spn_client_secret: Service Principal Secret.

        :returns: ServicePrincipalAuthentication credentials
        :rtype: ServicePrincipalAuthentication
        """
        return ServicePrincipalCredentials(
            client_id=client_id,
            secret=secret_id,
            tenant=tenant_id,
            _enable_caching=False
        )

    @staticmethod
    def get_ds_object(ws, name):
        """
        get_ds_object - Get workspace datastore object

        :param str ws: workspace
        :param str name: data store name

        :returns: ws, name
        :rtype: blob object, str

       """
        return Datastore.get(ws, name)

    @staticmethod
    def get_compute_object(ws, compute_name, size="STANDARD_NC6", min_nodes=1, max_nodes=4):
        """
        get_compute_object - Retrieves a AMLS compute object.

        :param Workspace ws: AMLS Workspace object.
        :param str compute_name: AMLS compute name.

        :returns: MLS compute target
        :rtype: azureml.core.compute.ComputeTarget
        """
        if compute_name in ws.compute_targets:
            compute_target = ws.compute_targets[compute_name]

        else:
            provisioning_config = AmlCompute.provisioning_configuration(vm_size=size,
                                                                        min_nodes=min_nodes,
                                                                        max_nodes=max_nodes)
            # Create the cluster
            compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
            compute_target.wait_for_completion(
                show_output=True
            )
        return compute_target

    @staticmethod
    def get_run_cfg(ws, pip_packages, conda_packages, ext_wheels, gpu=True):
        '''
        get_run_cfg - Retrieves the AMLS run configuration.


        :returns: AMLS run configuration
        :rtype: RunConfiguration object
        '''
        conda_dep = CondaDependencies()
        for pip_package in pip_packages:
            conda_dep.add_pip_package(pip_package)
        for conda_package in conda_packages:
            conda_dep.add_conda_package(conda_package)
        for whl_path in ext_wheels:
            whl_url = Environment.add_private_pip_wheel(
                workspace=ws,
                file_path=whl_path,
                exist_ok=True
            )
            conda_dep.add_pip_package(whl_url)
        run_cfg = RunConfiguration(conda_dependencies=conda_dep)
        run_cfg.environment.docker.enabled = True
        run_cfg.environment.docker.gpu_support = gpu
        if gpu:
            run_cfg.environment.docker.base_image = DEFAULT_GPU_IMAGE
        else:
            run_cfg.environment.docker.base_image = DEFAULT_CPU_IMAGE
        run_cfg.environment.spark.precache_packages = False
        return run_cfg


def register_data_store(work_space, data_store_name, container_name, blob_account_name,
                        blob_account_key, set_default=False):
    """
    register_data_store - register datastore

    :param str data_store_name: workspace
    :param str container_name: data store name
    :param str blob_account_name: data store name
    :param str blob_account_key: data store name

    :returns: data_store
    :rtype: data store object

    """
    data_store = Datastore.register_azure_blob_container(
        workspace=work_space,
        datastore_name=data_store_name,
        container_name=container_name,
        account_name=blob_account_name,
        account_key=blob_account_key,
        create_if_not_exists=True)
    # Set it to default data store for the AML workspace
    if set_default:
        work_space.set_default_datastore(data_store_name)
    return data_store
