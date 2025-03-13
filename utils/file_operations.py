import json
from PyPDF2 import PdfReader

def read_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    docs = [page.extract_text() for page in reader.pages]
    return "\n\n\n --- \n\n\n".join(docs)

def read_python_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def save_solution(solution, output_path="solution.json"):
    #solution = solution["generation"]
    solution_dict = {
        "description": solution.description,
        "imports": solution.imports,
        "executable_code": solution.executable_code,
    }
    
    with open(output_path, "w") as file:
        json.dump(solution_dict, file)

    with open(output_path.replace('.json', '.txt'), "w") as file:
        file.write(solution.description + "\n" + solution.executable_code) 