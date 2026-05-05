# Azure Deployment

This project is a Dockerized command-line workflow. The simplest Azure target is
Azure Container Instances because the container can run the watermarking script,
write logs, then stop.

## Prerequisites

- Azure CLI installed locally
- Docker image builds successfully locally
- An Azure subscription
- You are logged in with:

```bash
az login
```

## 1. Choose Names

Azure Container Registry names must be globally unique and use only lowercase
letters and numbers.

```bash
$RESOURCE_GROUP="qim-watermark-rg"
$LOCATION="francecentral"
$ACR_NAME="qimwatermark$((Get-Random -Maximum 99999))"
$IMAGE_NAME="qim-watermark-project"
$TAG="v1"
$CONTAINER_NAME="qim-watermark-run"
```

## 2. Create Azure Resources

```bash
az group create `
  --name $RESOURCE_GROUP `
  --location $LOCATION
```

```bash
az acr create `
  --resource-group $RESOURCE_GROUP `
  --name $ACR_NAME `
  --sku Basic
```

## 3. Build And Push The Image To Azure

This builds the Docker image in Azure Container Registry from the current
repository directory.

```bash
az acr build `
  --registry $ACR_NAME `
  --image "$IMAGE_NAME:$TAG" `
  .
```

## 4. Deploy A One-Time Container Run

For a simple student/demo deployment, enable the registry admin account and use
its generated credentials for Azure Container Instances.

```bash
az acr update `
  --name $ACR_NAME `
  --admin-enabled true
```

```bash
$ACR_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
$ACR_USER=$(az acr credential show --name $ACR_NAME --query username --output tsv)
$ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" --output tsv)
```

```bash
az container create `
  --resource-group $RESOURCE_GROUP `
  --name $CONTAINER_NAME `
  --image "$ACR_SERVER/$IMAGE_NAME:$TAG" `
  --registry-login-server $ACR_SERVER `
  --registry-username $ACR_USER `
  --registry-password $ACR_PASSWORD `
  --restart-policy Never `
  --cpu 1 `
  --memory 2
```

## 5. Check Logs

```bash
az container logs `
  --resource-group $RESOURCE_GROUP `
  --name $CONTAINER_NAME
```

Check the final state:

```bash
az container show `
  --resource-group $RESOURCE_GROUP `
  --name $CONTAINER_NAME `
  --query "containers[0].instanceView.currentState"
```

## 6. Clean Up

When you no longer need the Azure resources:

```bash
az group delete `
  --name $RESOURCE_GROUP `
  --yes
```

## Notes

- The current Dockerfile runs with `test_assets/host.png` inside the image.
- For production, prefer a service principal or managed identity instead of the
  registry admin account.
- If you want to process user-uploaded images in Azure, the next step is adding
  Azure Blob Storage input/output paths.

## References

- Azure Container Instances documentation:
  https://learn.microsoft.com/en-us/azure/container-instances/
- Azure Container Registry quickstart:
  https://learn.microsoft.com/en-us/azure/container-registry/container-registry-get-started-azure-cli
- Deploy Azure Container Instances from Azure Container Registry:
  https://learn.microsoft.com/en-us/azure/container-instances/container-instances-using-azure-container-registry
