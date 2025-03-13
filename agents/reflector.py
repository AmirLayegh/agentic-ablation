from langchain_openai import ChatOpenAI
from models.schemas import Code, GraphState
from prompts.templates import REFLECTION_PROMPT

class ReflectorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.chain = REFLECTION_PROMPT | self.llm.with_structured_output(Code)

    def run(self, state: GraphState):
        print("-----Reflecting on Code Solution-----")
        
        messages = state["messages"]
        code_solution = state["generation"]
        iterations = state["iterations"]
        
        reflections = self.chain.invoke({"messages": messages})
        messages += [
            (
                "assistant",
                f"Here are reflections on the error: {reflections.description} \n Imports and Executable Code: {reflections.executable_code}"
            )
        ]
        return {
            "generation": reflections.executable_code,
            "messages": messages,
            "iterations": iterations,
        } 