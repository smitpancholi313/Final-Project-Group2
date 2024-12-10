import streamlit as st
import pinecone
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from groq import Groq
from pinecone import Pinecone, ServerlessSpec


def get_relevant_excerpts(user_question, docsearch):
    relevent_docs = docsearch.similarity_search(user_question)
    return '\n\n------------------------------------------------------\n\n'.join(
        [doc.page_content for doc in relevent_docs[:3]]
    )


def presidential_speech_chat_completion(client, model, user_question, relevant_excerpts, additional_context):
    system_prompt = '''
    You are a presidential historian. Given the user's question and relevant excerpts from 
    presidential speeches, answer the question by including direct quotes from presidential speeches. 
    When using a quote, cite the speech that it was from (ignoring the chunk).
    '''

    if additional_context:
        system_prompt += f"\nThe user has provided this additional context:\n{additional_context}"

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Question: {user_question}\n\nRelevant Speech Excerpt(s):\n\n{relevant_excerpts}",
            },
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content


def main():
    st.write(st.secrets)

    groq_api_key = st.secrets["secrets"]["GROQ_API_KEY"]
    pinecone_api_key = st.secrets["secrets"]["PINECONE_API_KEY"]
    # pinecone_index_name = "presidential-speeches"

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists
    index_name = "presidential-speeches"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust to match the embedding model dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # Initialize PineconeVectorStore
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embedding_function)

    client = Groq(api_key=groq_api_key)

    st.image('groqcloud_darkmode.png')
    st.title("Presidential Speeches RAG")
    st.markdown("""
        Welcome! Ask questions about U.S. presidents, like "What were George Washington's views on democracy?" or "What did Abraham Lincoln say about national unity?". 
        The app matches your question to relevant excerpts from presidential speeches and generates a response using a pre-trained model.
    """)

    additional_context = st.sidebar.text_input('Additional summarization context (optional):')
    model = st.sidebar.selectbox('Choose a model', ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'])
    user_question = st.text_input("Ask a question about a US president:")

    if user_question:
        relevant_excerpts = get_relevant_excerpts(user_question, docsearch)
        response = presidential_speech_chat_completion(client, model, user_question, relevant_excerpts, additional_context)
        st.write(response)


if __name__ == "__main__":
    main()
