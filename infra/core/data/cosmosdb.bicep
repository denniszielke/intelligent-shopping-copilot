metadata description = 'Create an Azure Cosmos DB account.'

param name string
param location string = resourceGroup().location
param tags object = {}

@allowed(['GlobalDocumentDB', 'MongoDB', 'Parse'])
@description('Sets the kind of account.')
param kind string = 'MongoDB'

@description('Enables serverless for this account. Defaults to false.')
param enableServerless bool = true

@description('Enables NoSQL vector search for this account. Defaults to false.')
param enableNoSQLVectorSearch bool = false

@description('Disables key-based authentication. Defaults to false.')
param disableKeyBasedAuth bool = false

resource account 'Microsoft.DocumentDB/databaseAccounts@2024-05-15' = {
  name: name
  location: location
  kind: kind
  tags: tags
  properties: {
    consistencyPolicy: {
      defaultConsistencyLevel: 'Session'
    }
    databaseAccountOfferType: 'Standard'
    locations: [
      {
        locationName: location
        failoverPriority: 0
        isZoneRedundant: false
      }
    ]
    publicNetworkAccess: 'Enabled'
    enableAutomaticFailover: false
    enableMultipleWriteLocations: false
    apiProperties: (kind == 'MongoDB')
      ? {
          serverVersion: '7.0'
        }
      : {}
    disableLocalAuth: disableKeyBasedAuth
    capabilities: union(
      (enableServerless)
        ? [
            {
              name: 'EnableServerless'
            }
          ]
        : [],
      (kind == 'MongoDB')
        ? [
            {
              name: 'EnableMongo'
            }
          ]
        : [],
      (enableNoSQLVectorSearch)
        ? [
            {
              name: 'EnableNoSQLVectorSearch'
            }
          ]
        : []
    )
  }
}

output endpoint string = account.properties.documentEndpoint
output name string = account.name
output connectionString string = account.listConnectionStrings().connectionStrings[0].connectionString
