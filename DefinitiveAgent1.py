from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

class CodeGenState(TypedDict):
    """

    messages : Everything that has been said so far in the conversation
    code : Code solution

    """
    messages: List
    code: str


def code_generate(state: CodeGenState):

    messages = state["messages"]


    code_solution = code_gen_complete.invoke(
        {"messages": messages, 
         "java_code": java_code,
        "test_code": test_code}
    )
    messages += [
        (
            "assistant",
            f"{code_solution.description} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    return {"code": code_solution, "messages": messages}


code_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in java tests for the specific project\n 
            Answer the user 
            question based on the above provided documentation. Ensure any code you provide can be executed \n 
            with all required imports and variables defined. Structure your answer with a description of the code solution. \n
            Then list the imports. And finally list the functioning code block as it could be normally executed. \n
            You need to find which inputs, already visible in the test case, or which methods are those that are most important if invoked differently with other types of inputs 
            (most impactful methods/inputs that if changed can increase the coverage of the test case).\n
            Here is the java code: {java_code} \n
            Here is the test code you need to improve coverage: {test_code} \n
            Here is the user question:""",
        ),
        ("placeholder", "{messages}"),
    ]
)

class code(BaseModel):

    description: str = Field(description="Description of what needs to be done to improve test coverage")
    imports: str = Field(description="Code imports")
    code: str = Field(description="Improved code not including imports")

nome_file = "dati.txt"  # Nome del file di input
# Se il percorso è solo il nome del file, lo cerca nella cartella dello script
if not os.path.isabs(nome_file):
    nome_file = os.path.join(os.getcwd(), nome_file)  # Lo converte in percorso assoluto

# Check if the file exist
if not os.path.exists(nome_file):
    with open(nome_file, "w", encoding="utf-8") as file:
        file.write("File creato perché non esisteva.\n")
    print(f"File '{nome_file}' creato con successo!")

#Read del codice dal file
try:
    if not os.path.exists(nome_file):
        print(f"File non trovato: {nome_file}")
        exit()
    with open(nome_file, "r") as f:
        file_content = f.read()
except PermissionError:
    print(f"Permessi negati per il file: {nome_file}")
    exit()
except Exception as e:
    print(f"Errore durante la lettura del file: {e}")
    exit()
# Separazione del metodo Java e dei casi di test
java_code = "" # Inizializza la variabile per il codice Java
test_code = "" # Inizializza la variabile per i casi di test
lines = file_content.splitlines()  # Inizializza la lista per le righe del file

for x in range(0, len(lines)):
    i=x
    line=lines[i]
    if line.strip() == "---JAVA---":
        java_code = ""
    elif line != "---TEST---":
        java_code += line + "\n"
    elif line.strip()== "---TEST---":
        break
for x in range(i+1, len(lines)):
    line=lines[x]
    test_code += line + "\n"




llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash") #gemini-2.5-pro-exp-03-25")   #"gemini-2.0-flash")

code_gen_complete = code_prompt | llm.with_structured_output(code)

graph = StateGraph(CodeGenState)

graph.add_node("generate", code_generate)
graph.add_edge(START, "generate")
graph.add_edge("generate", END)

app = graph.compile()


question = "You need to work on improving the test coverage of the code provided. \n"
solution = app.invoke({"messages": [("user", question)],})

print(solution)