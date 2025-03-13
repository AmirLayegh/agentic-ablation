from models.schemas import GraphState

class ExecutorAgent:
    def run(self, state: GraphState):
        print("-----Testing Code Solution-----")        
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]
        
        try:
            exec(code_solution.imports, globals())
        except Exception as e:
            print("======Code Import Error=======")
            error_message = [("user", f"Your solution failed the import test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }
            
        try:
            exec(code_solution.executable_code, globals())
        except Exception as e:
            print("---CODE BLOCK CHECK: FAILED---")
            error_message = [("user", f"Your solution failed the code execution test: {e}")]
            messages += error_message
            return {
                "generation": code_solution,
                "messages": messages,
                "iterations": iterations,
                "error": "yes",
            }

        print("---NO CODE TEST FAILURES---")
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "no",
        } 