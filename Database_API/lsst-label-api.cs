
using System;
using System.Data;
using System.Data.SqlClient;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.WindowsAzure.Storage;
using Microsoft.WindowsAzure.Storage.Auth;
using Microsoft.WindowsAzure.Storage.Blob;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Azure.WebJobs.Host;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;

namespace Database_API
{
    public static class lsst_label_api
    {
        [FunctionName("lsst-label-api")]
        public static async Task<IActionResult> Run([HttpTrigger(AuthorizationLevel.Function, "post", "get", Route = null)]HttpRequest req, ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            // setup storage credentials
            StorageCredentials storageCredentials = new StorageCredentials("lsstlabelstorage", "rOGgsGcMEmRPujiKnfghioCeCgNOPgQYWIROlWO8jK/M/hTjrKiJxc3mb12lIuw0QHxILOBV8cD8N/i+/K+5/g==");
            CloudStorageAccount storageAccount = new CloudStorageAccount(storageCredentials, true);
            CloudBlobClient blobClient = storageAccount.CreateCloudBlobClient();
            CloudBlobContainer blobContainer = blobClient.GetContainerReference("lsst-image-container");

            SqlConnectionStringBuilder builder = new SqlConnectionStringBuilder();
            builder.DataSource = "lsst-label-server.database.windows.net";
            builder.UserID = "lsst-label-admin";
            builder.Password = "QtGxJ/;T[8qx";
            builder.InitialCatalog = "LSST-Label-Database";

            if (req.Method.Equals("GET"))
            {
                String response = "";

                using (SqlConnection connection = new SqlConnection(builder.ConnectionString))
                {
                    connection.Open();

                    StringBuilder sb = new StringBuilder();

                    sb.Append("SELECT TOP 1 URL FROM dbo.IMAGE_DATA");
                    String querySql = sb.ToString();

                    using (SqlCommand command = new SqlCommand(querySql, connection))
                    {
                        using (SqlDataReader reader = command.ExecuteReader())
                        {
                            while (reader.Read())
                            {
                                response += reader.GetString(0);
                            }
                        }
                    }
                }

                return (ActionResult) new OkObjectResult(response);
            }
            else if (req.Method.Equals("POST"))
            {
                // read in image bytes from the request body and store in file
                using (BinaryReader reader = new BinaryReader(req.Body)) {
                    Byte[] lnByte = reader.ReadBytes(1 * 1024 * 1024 * 10);
                    using (FileStream imageFileStream = new FileStream("uploadImage.jpg", FileMode.Create)) {
                        imageFileStream.Write(lnByte, 0, lnByte.Length);
                    }
                }

                // upload image to the blob
                CloudBlockBlob blockBlob = blobContainer.GetBlockBlobReference("uploadedImage.jpg");
                await blockBlob.UploadFromFileAsync("uploadImage.jpg");

                bool success = await Task.FromResult(true);
                return (ActionResult) new OkObjectResult("Congrats! You've attempted a POST! Check the database and Blob Storage to see if it worked! Success?: " + success);
            }

            return (ActionResult) new OkObjectResult("Sorry! " + req.Method + " actions not currently supported!");
        }
    }
}