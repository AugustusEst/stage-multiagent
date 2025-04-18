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


llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash")
def agent_1(state: MessagesState) -> Command[Literal["agent_2", END]]:
    

    def parse_java_code(java_code):
        try:
            tree = javalang.parse.parse(java_code)
            class_names = [node.name for _, node in tree.filter(ClassDeclaration)]
            return str(class_names)
        except Exception as e:
            return f"Errore durante l'analisi: {e}"
        
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
    result = f"### Risultato dell'analisi per test case ###\n{analysis.content}"
    new_messages.append(HumanMessage(content=result, name="agent_1"))

    return Command(
    goto="agent_2",
    update=new_messages,
    )
       
    

def agent_2(state: MessagesState) -> Command[Literal["agent_1", END]]:
    print("---AGENT 2: CODE GENERATION AGENT---") 

    # Code generation agent setup
    class CodeGenState(TypedDict):
        """State for the code generation agent"""
        error: str
        messages: List
        generation: str
        iterations: int
    
    

    class code(BaseModel):
        """Schema for code solutions"""
        prefix: str = Field(description="Description of the problem and approach")
        imports: str = Field(description="Code block import statements")
        code: str = Field(description="Code block not including import statements")
    
    # Set up context from a dummy URL for demonstration
    # In a real scenario, this would be retrieved based on the query
    concatenated_content = "Example context for code generation"
    
    # LLM for code generation with structured output
    
    # Code generation prompt
    code_gen_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer the user question based on your knowledge.
            Ensure any code you provide can be executed with all required imports and variables defined. 
            Structure your answer with a description of the code solution, imports, and functioning code block.
            You need to find which inputs, already visible in the test case, or which methods are those that are most important if invoked differently with other types of inputs 
            (most impactful methods/inputs that if changed can increase the coverage of the test case).
            Java code: \n{java_code}\n\n
            Test code you need to improve: \n{test_code}\n\n"""
        ),
        ("placeholder", "{messages}"),
    ])
    
    code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
    
    # Max tries and reflection flag
    max_iterations = 0
    flag = "do not reflect"
    
    # Generate code function
    def generate(state: CodeGenState):
        print("---GENERATING CODE SOLUTION---")
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        
        """if error == "yes":
            messages += [(
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"
            )]"""
        
        # Extract the latest user message from the MessagesState
        user_query = ""
        for msg in messages:
            if msg[0] == "user":
                user_query = msg[1]
        
        # Solution
        code_solution = code_gen_chain.invoke({
            "java_code": java_code,
            "test_code": test_code,
            "context": concatenated_content, 
            "messages": [("user", user_query)]
        })
        
        messages += [(
            "assistant",
            f"{code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}"
        )]
        
        # Increment
        iterations = iterations + 1
        return {"generation": code_solution, "messages": messages, "iterations": iterations}
    
    

    # Code check function
    def code_check(state: CodeGenState):
        print("---CHECKING CODE---")
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]
        
        # Get solution components
        imports = code_solution.imports
        code = code_solution.code
        
        # Simple validation - in real scenario we'd execute the code
        """if "import" not in imports.lower():
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: No imports found")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }"""
        
        # No errors
        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        }
    
    # Decision function
    def decide_to_finish(state: CodeGenState):
        error = state["error"]
        iterations = state["iterations"]
        
        #if error == "no" or iterations == max_iterations:
        print("---DECISION: FINISH---")
        return "end"
        """else:
            print("---DECISION: RE-TRY SOLUTION---")
            return "generate"""
    
    # Create the code generation workflow
    workflow = StateGraph(CodeGenState)
    
    # Define the nodes
    workflow.add_node("generate", generate)
    workflow.add_node("check_code", code_check)
    
    # Build graph
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "check_code")
    workflow.add_conditional_edges(
        "check_code",
        decide_to_finish,
        {
            "end": END,
            "generate": "generate",
        },
    )
    
    app = workflow.compile()
    
    # Extract the user query from the agent messages
    user_query = ""
    for msg in state["messages"]:
        if isinstance(msg, tuple) and msg[0] == "user":
            user_query = msg[1]
        elif hasattr(msg, 'content'):
            if msg.type == "human":
                user_query = msg.content
    
    # Run the code generation workflow
    solution = app.invoke({
        "messages": [("user", user_query)], 
        "iterations": 0, 
        "error": ""
    })
    
    # Create a response message with the code solution
    if "generation" in solution and solution["generation"]:
        code_sol = solution["generation"]
        final_response = (
            f"FINAL ANSWER\n\nHere's the code solution:\n\n"
            f"{code_sol.prefix}\n\n"
            f"Imports:\n```python\n{code_sol.imports}\n```\n\n"
            f"Code:\n```python\n{code_sol.code}\n```"
        )
    else:
        final_response = "FINAL ANSWER\n\nI couldn't generate a proper code solution."
    
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






"""
events = network.stream(
    {
        "messages": [
            (
                "user",
                "First, comment the test code, then"
                "You need to find which inputs, already visible in the test case, or which methods are those that are most important if invoked differently with other types of inputs "
                "(most impactful methods/inputs that if changed can increase the coverage of the test case). "
                "Once you make it, finish.",
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"recursion_limit": 10},
)

for s in events:
    print(s)
    print("----") 
    """