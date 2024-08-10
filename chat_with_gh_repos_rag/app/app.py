# Initialize Qdrant Vector Store
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embed_model=embed_model)

# Initialize Vector Store Index
index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)

# Define the prompt template for querying
qa_prompt_tmpl_str = """\
Context information is below.
---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, \
answer the query. Please be concise, and complete. \
If the context does not contain an answer to the query \
respond with I don't know!

Query: {query_str}
Answer: \
"""
qa_prompt = PromptTemplate(qa_prompt_tmpl_str)

# Initialize Retriever
retriever = VectorIndexRetriever(index=index)

# Initialize Response Synthesizer
response_synthesizer = get_response_synthesizer(
    text_qa_template=qa_prompt,
)

# Initialize Sentence Reranker for query response
rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

# Initialize RetrieverQueryEngine for query processing
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[rerank]
)

@app.post("/query/")
async def query_vector_store(request: QueryRequest):

    query = request.query
    response = query_engine.query(query)
    if not response:
        raise HTTPException(status_code=404, detail="No response found")

    # Remove newline characters from the response
    cleaned_response = response.response.replace("\n", "")

    return cleaned_response