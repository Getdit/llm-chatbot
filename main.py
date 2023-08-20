import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embed = True
retriever = None

config = {
        'file_path': './Effective-Python.pdf',
        'start_page': 23,
        'end_page': 63,
        'question': input("Type your question: "),
    }

if embed:
    #------------------- LOADING THE DOCUMENT -------------------#
    from langchain.document_loaders import PyPDFLoader

    loader = PyPDFLoader(file_path=config["file_path"])
    pages = loader.load()
    logger.info(f"Data loaded successuflly: {len(pages)} pages")
    filtered_pages = [
        page
        for page in pages
        if config["start_page"] <= page.metadata["page"] <= config["end_page"]
    ]
    logger.info(f"Filtred page in the following rand: [{config['start_page']}, {config['end_page']}]")

    #------------------- end LOADING THE DOCUMENT -------------------#

    #------------------- SPLITTING PAGES IN CHUNKS ------------------#
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = text_splitter.split_documents(filtered_pages)
    logger.info(f"Splitted documents into {len(chunks)} chunks")

    #----------------- end SPLITTING PAGES IN CHUNKS ----------------#

    #------------------- EMBED THE CHUNKS AND INDEX TO CHROMA ----------------------#
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OpenAIEmbeddings

    logger.info("Embedding the chunks and indexing them into Chroma ...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory="db",
    )
    logger.info("Chunks indexed into Chroma")
    #----------------- end EMBED THE CHUNKS AND INDEX TO CHROMA --------------------#

#------------------- ASKING A QUESTION ----------------------#
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)
answer = qa_chain({"query": config['question']})
print(answer["result"])
#----------------- end ASKING A QUESTION --------------------#