from langchain_openai import ChatOpenAI
from models.schemas import Code, GraphState
from prompts.templates import CODE_GEN_PROMPT

class CodeGeneratorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="o3-mini")
        self.chain = CODE_GEN_PROMPT | self.llm.with_structured_output(Code)

    def run(self, state: GraphState):
        print("-----Generating Code Solution-----")
        
        messages = state["messages"]
        iterations = state["iterations"]
        error = state["error"]
        
        if error == "yes":
            messages += [
                (
                    "user",
                    "Now, try again since the previous code solution was not correct. Invoke the code tool to structure output with a description, imports, and ablated code block."
                )
            ]
            
        code_solution = self.chain.invoke({"messages": messages})
        messages += [
            (
                "assistant",
                f"{code_solution.description} \n Imports: {code_solution.imports} \n Executable Code: {code_solution.executable_code}"
            )
        ]
        iterations += 1
        
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
        } 