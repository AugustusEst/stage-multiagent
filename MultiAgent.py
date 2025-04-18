from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any
from typing_extensions import TypedDict
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph, START, MessagesState
from langgraph.types import Command
from langchain_core.messages import convert_to_messages
import javalang
from javalang.tree import ClassDeclaration
from langchain_core.prompts import PromptTemplate

# Code generation related imports
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

nome_file = "dati.txt"  # Nome del file di input
# Se il percorso è solo il nome del file, lo cerca nella cartella dello script
if not os.path.isabs(nome_file):
    nome_file = os.path.join(os.getcwd(), nome_file)  # Lo converte in percorso assoluto

# Check if the file exist
if not os.path.exists(nome_file):
    with open(nome_file, "w", encoding="utf-8") as file:
        file.write("File created because it did not exist.\n")
    print(f"File '{nome_file}' successfully created!")

#Read del codice dal file
try:
    if not os.path.exists(nome_file):
        print(f"File not found: {nome_file}")
        exit()
    with open(nome_file, "r") as f:
        file_content = f.read()
except PermissionError:
    print(f"Permissions denied for the file: {nome_file}")
    exit()
except Exception as e:
    print(f"Error while reading the file: {e}")
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


llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash")
def agent_1(state: MessagesState) -> Command[Literal["agent_2", END]]:
    

    def parse_java_code(java_code):
        try:
            tree = javalang.parse.parse(java_code)
            class_names = [node.name for _, node in tree.filter(ClassDeclaration)]
            return str(class_names)
        except Exception as e:
            return f"Error during analysis: {e}"
        
    def generate_prompt(java_code, test_code, parsed_code):
        template = PromptTemplate(
            input_variables=["java_code", "test_code", "parsed_code"],
            template=(
                "I provide you with the following Java method and its associated test cases:\n\n"
                "Java Code:\n{java_code}\n\n"
                "Test Code:\n{test_code}\n\n"
                "You must provide me with a detailed description of the functionality tested by the test cases (example: test01 tests functionality...),\n\n"
                "then you must also explain to me what the test cases specifically test, referring directly to the relevant parts of code.\n\n"
                "I want you to provide me with an orderly and non-generic answer as if I were to pass this information on to someone who then needs it to improve test case coverage (I don't need a method description).\n\n"
            )
        )
        return llm.invoke(template.format(java_code=java_code, test_code=test_code, parsed_code=parsed_code))
        
    parsed_code = parse_java_code(java_code)
    new_messages = state["messages"]

    analysis = generate_prompt(java_code, test_code, parsed_code)
    result = f"### Analysis result per test case ###\n{analysis.content}"
    new_messages.append(HumanMessage(content=result, name="agent_1"))

    return Command(
    goto="agent_2",
    update=new_messages,
    )
       
    

def agent_2(state: MessagesState) -> Command[Literal["agent_1", END]]:
    
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



    llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash") #gemini-2.5-pro-exp-03-25")   #"gemini-2.0-flash")

    code_gen_complete = code_prompt | llm.with_structured_output(code)

    graph = StateGraph(CodeGenState)

    graph.add_node("generate", code_generate)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)

    app = graph.compile()

    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, tuple) and msg[0] == "user":
            user_query = msg[1]
        elif hasattr(msg, 'content'):
            if msg.type == "human":
                user_query = msg.content

    solution = app.invoke({"messages": [("user", user_query)],})
    code_sol = solution["code"]
    final_response = (
            f"FINAL ANSWER\n\nHere's the code solution:\n\n"
            f"{code_sol.description}\n\n"
            f"Imports:\n```\n{code_sol.imports}\n```\n\n"
            f"Code:\n```\n{code_sol.code}\n```"
        )

    
    # Add the response to the state
    response = {"messages": state["messages"] + [HumanMessage(content=final_response, name="agent_2")]}
    
    return Command(
        goto=END,
        update=response,
    )

builder = StateGraph(MessagesState)
builder.add_node(agent_1)
builder.add_node(agent_2)

builder.add_edge(START, "agent_1")
network = builder.compile()

def pretty_print_messages(update):
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:")
        print("\n")

        for m in convert_to_messages(node_update["messages"]):
            m.pretty_print()
        print("\n")

final_state = network.invoke({
    "messages": [("user", 
        "Imagine you are an expert analyst of Java code and unit tests.\n\n"
        "I hand you the Java method of a class and its associated test cases.\n\n"
        "You will have to analyse them carefully and provide me with an answer according to the question I ask you...\n\n"
        "I want you to give me a clear and straight answer to the question I ask you. \n\n"
    )],
})

for m in convert_to_messages(final_state["messages"]):
    m.pretty_print()
