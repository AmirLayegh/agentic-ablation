from langgraph.graph import END, StateGraph, START
from agents import CodeGeneratorAgent, ExecutorAgent, ReflectorAgent, TerminatorAgent, AnalyzerAgent
from models import GraphState

class AblationWorkflow:
    def __init__(self, max_iterations=3, reflection_enabled=True):
        self.max_iterations = max_iterations
        self.reflection_enabled = reflection_enabled
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        # Initialize agents
        code_generator = CodeGeneratorAgent()
        executor = ExecutorAgent()
        reflector = ReflectorAgent()
        analyzer = AnalyzerAgent()

        # Create workflow graph
        workflow = StateGraph(GraphState)
        
        workflow.add_node("code_generator", code_generator.run)
        workflow.add_node("executor", executor.run)
        if self.reflection_enabled:
            workflow.add_node("reflector", reflector.run)
        workflow.add_node("analyzer", analyzer.run)
        
        # Add edges
        workflow.add_edge(START, "code_generator")
        workflow.add_edge("code_generator", "executor")
        workflow.add_conditional_edges(
            "executor",
            TerminatorAgent().run,
            {
                "analyzer": "analyzer",
                "reflector": "reflector",
            }
        )
        workflow.add_edge("reflector", "code_generator")
        workflow.add_edge("analyzer", END)
        print(workflow)
        #workflow.compile()
        return workflow

    def run(self, initial_state):
        return self.workflow.invoke(initial_state)