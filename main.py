from langgraph.graph import END, StateGraph, START
from agents import (
    CodeGeneratorAgent,
    ExecutorAgent,
    ReflectorAgent,
    AnalyzerAgent
)
from models import GraphState
from utils.file_operations import save_solution
def decide_to_finish(state):
    error = state["error"]
    iterations = state["iterations"]
    if error == "no" or iterations >= 3:  # Max iterations check
        print("-----Finished-----")
        return "analyzer"
    else:
        print("----RE-TRY Solution-----")
        return "reflector"

def main():
    # Initialize agents
    code_generator_agent = CodeGeneratorAgent().run
    executor_agent = ExecutorAgent().run
    reflector_agent = ReflectorAgent().run
    analyzer_agent = AnalyzerAgent().run

    # Create workflow graph
    workflow = StateGraph(GraphState)
    
    # Add nodes and edges
    workflow.add_node("code_generator", code_generator_agent)
    workflow.add_node("executor", executor_agent)
    workflow.add_node("reflector", reflector_agent)
    workflow.add_node("analyzer", analyzer_agent)

    workflow.add_edge(START, "code_generator")
    workflow.add_edge("code_generator", "executor")
    workflow.add_conditional_edges(
        "executor",
        decide_to_finish,
        {
            "analyzer": "analyzer",
            "reflector": "reflector",
        },
    )
    workflow.add_edge("reflector", "code_generator")
    workflow.add_edge("analyzer", END)

    # Compile and visualize
    app = workflow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="ablation_study.png")

    # Read and process the code
    with open("/home/amir/agentic-ablation/cifar_cnn.py", "r") as file:
        original_code = file.read()
    
    # Run workflow
    solution = app.invoke({
        "messages": [("user", original_code)],
        "iterations": 0,
        "error": ""
    })
    
    print("Final solution:", solution["generation"])
    save_solution(solution["generation"])

if __name__ == "__main__":
    main() 