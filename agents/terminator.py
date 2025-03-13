from models.schemas import GraphState
from agents.analyzer import AnalyzerAgent
class TerminatorAgent:
    def run(self, state: GraphState):
        error = state["error"]
        iterations = state["iterations"]
        if error == "yes" or iterations == 3:
            print("-----Terminating-----")
            return "analyzer"
        else:
            print("-----RETRYING-----")
            return "reflector"
        
