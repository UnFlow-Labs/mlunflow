from abc import abstractmethod
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from unflow.core.unflow_types import RState, Transformation
from unflow.graph.compute_graph import ComputeGraph
from .llm import BaseLLM , TransformersLLM, LLMConfig, LlamaLLM
from unflow.core.code.utils import format_context_changes_for_llm , get_all_context_str

module_path = Path(__file__).parent
default_llm_config = LLMConfig()

def get_state_code(state: RState) -> str:
    """
    Get the source code of the procedure associated with the given state.
    """
    return state.procedure_source + "\n" + get_all_context_str(state.context_data)
def convert_graph_to_text(graph: ComputeGraph, state: RState) -> str:
    """
    Convert a graph  representation to a text format for LLM input.
    This is a placeholder implementation; you can customize it as needed.
    """
    # get transformations that are connected to the given state
    transformations = graph.query_transformations(to_state=state.name)
    graph_text = ""
    for i,transformation in enumerate(transformations):
        # graph_text += f"Transformation {i}: {transformation.name}\n"
        graph_text += f"  State: {transformation.state1.name} [{transformation.state1.status}]\n"
        graph_text += f"  Args Changes: {transformation.args_changes_str()}\n"
        graph_text += f"  Code Changes: {transformation.p_changes}\n"
        graph_text += f"  Context Changes: {format_context_changes_for_llm(transformation.context_changes)}\n\n"
    return graph_text

@dataclass
class Intent:
    name: str
    confidence: float
    metadata: Optional[dict] = None
    reason: Optional[str] = None
    confidence_reason: Optional[str] = None
    def __str__(self):
        return f"Intent(name={self.name}, confidence={self.confidence}, reason={self.reason}, \
        confidence_reason={self.confidence_reason}, metadata={self.metadata})"
    def __dict__(self):
        return {
            "name": self.name,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "reason": self.reason,
            "confidence_reason": self.confidence_reason,
        }
@dataclass
class OutcomeLabel:
    outcome: str
    confidence: float
    metadata: Optional[dict] = None
    reason: Optional[str] = None
    confidence_reason: Optional[str] = None
    def __str__(self):
        return f"OutcomeLabel(label={self.outcome}, confidence={self.confidence}, reason={self.reason}, \
        confidence_reason={self.confidence_reason}, metadata={self.metadata})"
    def __dict__(self):
        return {
            "label": self.label,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "reason": self.reason,
            "confidence_reason": self.confidence_reason,
        }

class IntentDetector():
    def __init__(self):
        pass
    @abstractmethod
    def detect_intent(self, state: RState) -> Intent:
        pass
    @abstractmethod
    def _set_graph(self, graph: ComputeGraph):
        pass

class LLMIntentDetector(IntentDetector):
    def __init__(self, llm: BaseLLM | TransformersLLM | LlamaLLM | None = None, model_name: str | None = None,
                 llm_config: LLMConfig | None = None):
        super().__init__()
        if llm is None:
            if model_name is None:
                raise ValueError("Either llm or model_name must be provided.")
            self.model_name = model_name
            self.llm = TransformersLLM(model_name=model_name)
            self.llm_config = llm_config or default_llm_config
            
        else:
            self.llm = llm # type: ignore
            self.llm_config = llm_config or default_llm_config
    
    def _set_graph(self, graph: ComputeGraph):
        self.graph = graph 

    def predict_outcome(self, state: RState) -> OutcomeLabel:
        with (module_path / "prompts" / "outcome_prediction.txt").open("r") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            graph = convert_graph_to_text(self.graph, state),
            code = get_state_code(state),
            args = state.args,
        )
        print(f"LLM Prompt:\n{prompt}\n")
        response = self.llm.generate(prompt, config=self.llm_config)
        print(f"LLM Response:\n{response}\n")
        # Here you would parse the response to extract the outcome label and confidence score.
        # For simplicity, let's assume the response is in the format "OutcomeLabel:ConfidenceScore"
        
        # parse the respone to get the json object with outcome label and confidence score
        try: 
            import json
            #clean response to remove any extra characters before the json object
            response = response[response.find("{"):response.rfind("}")+1]
            response_json = json.loads(response)
            outcome_label = response_json.get("outcome", "Unknown")
            confidence = float(response_json.get("confidence", 0.0))
            reason = response_json.get("reason", None)
            confidence_reason = response_json.get("confidence_reason", None)
            return OutcomeLabel(outcome=outcome_label, confidence=confidence, reason=reason, 
                                confidence_reason=confidence_reason)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON. Falling back to simple parsing.")
            print(f"LLM Response: {response}")
            return OutcomeLabel(outcome="Unknown", confidence=0.0)

    def detect_intent(self, state: RState) -> Intent:
        with (module_path / "prompts" / "general_json.txt").open("r") as f:
            prompt_template = f.read()
        prompt = prompt_template.format(
            graph = convert_graph_to_text(self.graph, state),
            code = state.procedure_source,
            args = state.args,
        )
        print(f"LLM Prompt:\n{prompt}\n")
        response = self.llm.generate(prompt, config=self.llm_config)
        print(f"LLM Response:\n{response}\n")
        # Here you would parse the response to extract the intent name and confidence score.
        # For simplicity, let's assume the response is in the format "IntentName:ConfidenceScore"
        
        # parse the respone to get the json object with intent name and confidence score
        try: 
            import json
            #clean response to remove any extra characters before the json object
            response = response[response.find("{"):response.rfind("}")+1]
            response_json = json.loads(response)
            intent_name = response_json.get("intent", "Unknown")
            confidence = float(response_json.get("confidence", 0.0))
            reason = response_json.get("reason", None)
            confidence_reason = response_json.get("confidence_reason", None)
            return Intent(name=intent_name, confidence=confidence, reason=reason, 
                          confidence_reason=confidence_reason)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON. Falling back to simple parsing.")
            print(f"LLM Response: {response}")
            return Intent(name="Unknown", confidence=0.0)