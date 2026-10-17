// Creates the dedicated workspace for this proof of concept.
// The subscription policy turns public network access off on storage
// accounts. This file does not try to fight that policy.
// Deploy:
//   az deployment group create --resource-group rg-rotorcraft-pdm --template-file infra/main.bicep

param location string = 'eastus'
param workspaceName string = 'mlw-rotorcraft-pdm'
param storageName string = 'strotorpdm001'

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowSharedKeyAccess: false
    publicNetworkAccess: 'Disabled'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kvrotor${uniqueString(resourceGroup().id)}'
  location: location
  properties: {
    tenantId: tenant().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    publicNetworkAccess: 'Enabled'
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-rotor-pdm'
  location: location
  kind: 'web'
  properties: { Application_Type: 'web' }
}

resource workspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: workspaceName
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    storageAccount: storage.id
    keyVault: keyVault.id
    applicationInsights: appInsights.id
    publicNetworkAccess: 'Enabled'
  }
}

output workspaceName string = workspace.name
output storageName string = storage.name
