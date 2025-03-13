from langchain_openai import ChatOpenAI
from models.schemas import Analysis, GraphState
from prompts.templates import ANALYZER_PROMPT

class AnalyzerAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="o3-mini")
        self.chain = ANALYZER_PROMPT | self.llm.with_structured_output(Analysis)

    def run(self, state: GraphState):
        print("-----Analyzing Results-----")
        messages = [
            ("user", 
             f"Here is the description of the ablated code: {state['generation'].description}\n"
             f"Here is the ablated code: {state['generation'].executable_code}\n"
             "Read the ablation_study_results.json file and analyze the results. "
             "Provide a summary of the results, including reasoning and analysis of the results."
            )
        ]
        
        analysis = self.chain.invoke({"messages": messages})
        messages += [
            (
                "assistant",
                f"Here is the analysis: {analysis.description}"
            )
        ]
        
        with open("analysis.txt", "w") as file:
            file.write(analysis.description)
            
        #return state 