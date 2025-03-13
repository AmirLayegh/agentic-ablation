from pydantic import BaseModel, Field
from typing import List
from typing_extensions import TypedDict

class Code(BaseModel):
    """A ablated code solution to a problem."""
    description: str = Field(description="A description of the code solution.")
    imports: str = Field(description="A list of imports.")
    executable_code: str = Field(description="A functioning code block.")

class Analysis(BaseModel):
    """An analysis of a result from an ablation study."""
    description: str = Field(description="A description of the analysis.")

class GraphState(TypedDict):
    """The state of the graph.
    Attributes:
    error: binary flag for control flow to indicate whether test error was tripped
    messages: With user question, error message, reasoning
    generation: Code solution
    iterations: Number of iterations
    """
    error: str
    messages: List
    generation: str
    iterations: int 