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

# Code generation related imports
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful coding assistant with expertise in java tests, collaborating with other code assistants."
        " Use the provided tools and your knowledge to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
        f"\n{suffix}"
    )

llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash")
def agent_1(state: MessagesState) -> Command[Literal["agent_2", END]]:
    agent_comments = create_react_agent(
    llm,
    tools=[],
    prompt=make_system_prompt(
        "You can only do maths. You are working with a research colleague."
    ),)
    response = agent_comments.invoke(state)
    response["messages"][-1] = HumanMessage(
        content=response["messages"][-1].content, name="agent_1"
    )
    print(response)
    return Command(
        goto= "agent_2",
        update={"messages": response["messages"]},
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
            """You are a coding assistant. Answer the user question based on your knowledge.
            Ensure any code you provide can be executed with all required imports and variables defined. 
            Structure your answer with a description of the code solution, imports, and functioning code block."""
        ),
        ("placeholder", "{messages}"),
    ])
    
    code_gen_chain = code_gen_prompt | llm.with_structured_output(code)
    
    # Max tries and reflection flag
    max_iterations = 3
    flag = "do not reflect"
    
    # Generate code function
    def generate(state: CodeGenState):
        print("---GENERATING CODE SOLUTION---")
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        
        if error == "yes":
            messages += [(
                "user",
                "Now, try again. Invoke the code tool to structure the output with a prefix, imports, and code block:"
            )]
        
        # Extract the latest user message from the MessagesState
        user_query = ""
        for msg in messages:
            if msg[0] == "user":
                user_query = msg[1]
        
        # Solution
        code_solution = code_gen_chain.invoke({
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
        if "import" not in imports.lower():
            print("---CODE IMPORT CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the import test: No imports found")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }
        
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
        
        if error == "no" or iterations == max_iterations:
            print("---DECISION: FINISH---")
            return "end"
        else:
            print("---DECISION: RE-TRY SOLUTION---")
            return "generate"
    
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