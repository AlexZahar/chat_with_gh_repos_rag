def get_code_file_list(github_token, github_username):

    try:
        # Initialize Github client
        g = Github(github_token)

        # Fetch all repositories for the user
        repos = g.get_user(github_username).get_repos()

        github_client = GithubClient(github_token=github_token, verbose=True)

        all_documents = []

        for repo in repos:
            repo_name = repo.full_name
            print(f"Loading files from {repo_name}")

            # Check if the repository belongs to the user
            if repo.owner.login != github_username:
                print(f"Skipping repository {repo_name} as it does not belong to the user.")
                continue

            try:
                # Determine the default branch
                default_branch = repo.default_branch

                # Load documents from the repository
                documents = GithubRepositoryReader(
                    github_client=github_client,
                    owner=github_username,
                    repo=repo.name,
                    use_parser=False,
                    verbose=False,
                    filter_file_extensions=(
                        [".py"],
                        GithubRepositoryReader.FilterType.INCLUDE,
                    ),
                ).load_data(branch=default_branch)

                # Ensure each document has text content
                for doc in documents:
                    if doc.text and doc.text.strip():
                        all_documents.append(doc)
                    else:
                        print(f"Skipping empty document: {doc.metadata['file_path']}")

            except Exception as e:
                print(f"Failed to load {repo_name}: {e}")

    except Exception as e:
        print(f"Error fetching repositories: {e}")

    return all_documents

def split_documents_into_nodes(all_documents):

    try:
        splitter = SentenceSplitter(
            chunk_size=1500,
            chunk_overlap=200
        )

        nodes = splitter.get_nodes_from_documents(all_documents)

        return nodes

    except Exception as e:
        print(f"Error splitting documents into nodes: {e}")
        return []

def create_collection_if_not_exists(client, collection_name):

    try:
        collections = client.get_collections()
        if collection_name not in [col.name for col in collections.collections]:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except ResponseHandlingException as e:
        print(f"Error checking or creating collection: {e}")


def chunked_nodes(data, client, collection_name):

    chunked_nodes = []

    for item in data:
        qdrant_id = str(uuid4())
        document_id = item.id_
        code_text = item.text
        source = item.metadata["url"]
        file_name = item.metadata["file_name"]

        content_vector = embed_model.get_text_embedding(code_text)

        payload = {
            "text": code_text,
            "document_id": document_id,
            "metadata": {
                            "qdrant_id": qdrant_id,
                            "source": source,
                            "file_name": file_name,
                            }
                }


        metadata = PointStruct(id=qdrant_id, vector=content_vector, payload=payload)

        chunked_nodes.append(metadata)

    if chunked_nodes:
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=chunked_nodes
        )

    print(f"{len(chunked_nodes)} Chunked metadata upserted.")