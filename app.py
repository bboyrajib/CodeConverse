from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
import getpass

openai_api_key = getpass.getpass("Enter your OpenAI API key: ")
os.environ["OPENAI_API_KEY"] = openai_api_key

activeloop_api_key = getpass.getpass("Enter your ActiveLoop API token: ")
os.environ["ACTIVELOOP_TOKEN"] = activeloop_api_key

embeddings = OpenAIEmbeddings()

username = input("Enter the ActiveLoop username: ")
database = input("Enter the existing DeepLake dataset path or an empty folder URL: ")
dataset_path = f"hub://{username}/{database}"

root_dir = input("Enter the root directory where your code exists: ")

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(texts)

db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

model = ChatOpenAI(model='gpt-3.5-turbo')  # switch to 'gpt-4'
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)

chat_history = []
question = ""

while question.lower() != "stop":
    question = input("Enter a question (or 'stop' to exit): ")
    if question.lower() == "stop":
        break
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
