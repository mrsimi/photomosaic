# import os, uuid
# from azure.identity import DefaultAzureCredential
# from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


# class BlobStorageUtil:
#     def __init__(self):
#         self.blob_service_client = BlobServiceClient.from_connection_string(blob_connection)
#         self.container_name = ""
#         self.container_client = ""

#     def create_container(self):
#         print('creating container')
#         self.container_name = str(uuid.uuid4())
#         self.container_client = self.blob_service_client.create_container(self.container_name, public_access='container')

#     def upload_single_file(self,file):
#         if file.filename != '':
#             blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file.filename)
#             blob_client.upload_blob(file)
#             # Construct the full URL of the uploaded file
#             blob_url = f"https://{blob_client.account_name}.blob.core.windows.net/{self.container_name}/{file.filename}"
#         return blob_url
    
#     def upload_multiple_files(self, files):
#         file_paths = [self.upload_single_file(file) for file in files]
#         return file_paths

#     def delete_container(self):
#         print('deleting blob container ...')
#         self.container_client.delete_container()

