from azure.common.credentials import ServicePrincipalCredentials
from azure.mgmt.storage import StorageManagementClient


class ResourceManager:
    """ Azure blob utilities for I/O operations
    """
    def __init__(self, spn_id, spn_secret, tenant_id):
        self.client_id = spn_id
        self.client_secret = spn_secret
        self.tenant_id = tenant_id

    def get_spn_credentials(self):
        self.credentials = ServicePrincipalCredentials(
            client_id=self.client_id,
            secret=self.client_secret,
            tenant=self.tenant_id,
            _enable_caching=False
        )

    def get_storage_account_key(self, account_name, subscription_id, rg_name):
        """
        :param str account_name: Azure Blob account name.
        :param ServicePrincipalCredential credentials: Service Principle Credentials.
        :param str subscription_id: Azure Subscription ID
        :param str rg_name: Azure Resource Group Name where storage account is listed

        """
        self.get_spn_credentials()
        storage_client = StorageManagementClient(self.credentials, subscription_id)
        storage_keys = storage_client.storage_accounts.list_keys(
            rg_name, account_name)
        storage_keys = {v.key_name: v.value for v in storage_keys.keys}
        account_key = storage_keys['key1']
        return account_key
