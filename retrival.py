import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    """it joins the document content and return back"""
    return "\n\n".join([doc.page_content for doc in docs])


promt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

embedding = OpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-small"
)

vector_store = PineconeVectorStore(
    index_name=os.environ.get("PINECONE_INDEX_NAME"), embedding=embedding
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini")


def retriever_without_lecl(query: str):
    docs = retriever.invoke(query)
    context = format_docs(docs)
    messages = promt_template.format_messages(context=context, question=query)
    response = llm.invoke(messages)
    return response.content


def create_retriever_chain_with_lecl():
    retrievr_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        )
        | promt_template
        | llm
        | StrOutputParser()
    )
    return retrievr_chain


if __name__ == "__main__":
    print("Retrieving...")

    query = "Tell me brief about burj khalifa?"

    # ========================================================================
    result_without_lcel = retriever_without_lecl(query)
    print("\nAnswer:")
    print(result_without_lcel)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 70)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 70)

    chain_with_lcel = create_retriever_chain_with_lecl()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print("\nAnswer:")
    print(result_with_lcel)
