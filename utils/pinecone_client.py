# /**
#  * This pinecone client is used in your api routes via the Frontend UI
#  */

from schema import CreatePineconeIndexArgs
import pinecone

# //use this client in api routes

async def create_pinecone_index(args: CreatePineconeIndexArgs):
    pinecone.init(api_key=args.pineconeApiKey,
                  environment=args.pineconeEnvironment)

    index = pinecone.Index(args.pineconeIndexName)

    print('index', index)

    return index
