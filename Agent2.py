from dotenv import load_dotenv
import javalang
from javalang.tree import ClassDeclaration
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Configura il modello LLM locale (Ollama)
#llm = ChatOllama(model="mistral", temperature=0.3, max_tokens=256)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model= "gemini-2.0-flash")

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
            "Imagine you are an expert analyst of Java code and unit tests.\n\n" 
                "I provide you with the following Java method and its associated test cases:\n\n"
                "Java Code:\n{java_code}\n\n"
                "Test Code:\n{test_code}\n\n"
                "You must provide me with a detailed description of the functionality tested by the test cases (example: test01 tests functionality...),\n\n"
                "then you must also explain to me what the test cases specifically test, referring directly to the relevant parts of code.\n\n"
                "I want you to provide me with an orderly and non-generic answer as if I were to pass this information on to someone who then needs it to improve test case coverage (I don't need a method description).\n\n"
        )
    )
    return llm.invoke(template.format(java_code=java_code, test_code=test_code, parsed_code=parsed_code))

# Programma principale
if __name__ == "__main__":
    print("____________________________________LLM Agent____________________________________")

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
    print(f"Permission denied for the file: {nome_file}")
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

    

print("\nOngoing analysis...\n")

try:
    parsed_code = parse_java_code(java_code)

    # Divide i test case per riga vuota o criterio tuo
    test_cases = test_code.strip().split("\n\n")  # oppure usa un separatore custom
    test_case_num = 1

    for test_case in test_cases:
        if test_case.strip():  # Salta test vuoti
            analysis = generate_prompt(java_code, test_case, parsed_code)
            print("="*80)
            print(f"\nAnalysis result per test case {test_case_num}:\n")
            print(test_case.strip())
            print("\nTest analysis:\n")
            print(analysis.content)
            print("="*80)
            test_case_num += 1

    print("\nAnalysis successfully completed!")
except Exception as e:
    print(f"Error during analysis: {e}")
