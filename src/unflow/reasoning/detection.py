from abc import abstractmethod
import ast
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from unflow.core.unflow_types import RState, Outcome
from unflow.graph.compute_graph import ComputeGraph
from unflow.core.builder import GraphBuilder
from .llm import BaseLLM , TransformersLLM, LLMConfig, LlamaLLM
from unflow.core.code.utils import format_context_changes_for_llm , get_all_context_str

module_path = Path(__file__).parent
default_llm_config = LLMConfig()

def get_state_code(state: RState) -> str:
    """
    Get the source code of the procedure associated with the given state.
    """
    return state.procedure_source + "\n" + get_all_context_str(state.context_data)


def check_python_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, "valid"
    except SyntaxError as err:
        line = err.lineno if err.lineno is not None else "unknown"
        offset = err.offset if err.offset is not None else "unknown"
        msg = err.msg if err.msg else "invalid syntax"
        return False, f"invalid: line={line}, col={offset}, message={msg}"


def _looks_like_syntax_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    return "syntax" in text and (
        "fail" in text or "error" in text or "won't run" in text or "cannot run" in text
    )


def _looks_like_hard_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    failure_terms = ["fail", "failure", "error", "crash", "cannot run", "won't run", "unsupported"]
    return any(term in text for term in failure_terms)


def _looks_like_env_device_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    env_terms = [
        "mps", "cuda", "gpu", "cpu", "device", "unsupported", "not available",
        "driver", "metal", "accelerator"
    ]
    failure_terms = ["fail", "error", "crash", "cannot run", "won't run", "unsupported"]
    return any(term in text for term in env_terms) and any(term in text for term in failure_terms)


def _has_explicit_env_device_evidence(state: RState, graph_text: str) -> bool:
    evidence_text = (get_state_code(state) + "\n" + str(state.args) + "\n" + graph_text).lower()
    evidence_terms = ["mps", "cuda", "gpu", "device", "torch.device", "metal", "accelerator"]
    return any(term in evidence_text for term in evidence_terms)


def _looks_like_unsupported_optimizer_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    return "optimizer" in text and "unsupported" in text and (
        "fail" in text or "error" in text or "cannot run" in text or "won't run" in text
    )


def _looks_like_optimizer_name_mismatch_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    mismatch_terms = ["mismatch", "typo", "wrong", "does not match", "discrepancy"]
    failure_terms = ["fail", "error", "cannot run", "won't run", "unsupported"]
    return "optimizer" in text and any(term in text for term in mismatch_terms) and any(
        term in text for term in failure_terms
    )


def _extract_claimed_optimizer_names(outcome: str, reason: str | None) -> list[str]:
    text = outcome + " " + (reason or "")
    patterns = [
        r"unsupported\s+optimizer\s*['\"`]([A-Za-z0-9_\-]+)['\"`]",
        r"optimizer\s*['\"`]([A-Za-z0-9_\-]+)['\"`]",
        r"uses\s*['\"`]([A-Za-z0-9_\-]+)['\"`]\s+as\s+the\s+optimizer",
    ]
    names: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            names.append(match.lower())
    unique_names = []
    for name in names:
        if name not in unique_names:
            unique_names.append(name)
    return unique_names


def _has_explicit_optimizer_claim_evidence(state: RState, graph_text: str, outcome: str, reason: str | None) -> bool:
    claimed_names = _extract_claimed_optimizer_names(outcome, reason)
    if not claimed_names:
        return False
    evidence_text = (str(state.args) + "\n" + graph_text).lower()
    return any(name in evidence_text for name in claimed_names)


def _extract_supported_optimizers_from_code(state: RState) -> set[str]:
    code = get_state_code(state)
    supported = set(
        match.lower() for match in re.findall(r"optimizer_name\s*==\s*['\"]([A-Za-z0-9_\-]+)['\"]", code)
    )
    return supported


def _extract_passed_optimizer_from_args(state: RState, graph_text: str) -> str | None:
    evidence_text = str(state.args) + "\n" + graph_text
    patterns = [
        r"['\"]optimizer['\"]\s*:\s*['\"]([A-Za-z0-9_\-]+)['\"]",
        r"optimizer\s*=\s*['\"]([A-Za-z0-9_\-]+)['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, evidence_text, flags=re.IGNORECASE)
        if match:
            return match.group(1).lower()
    return None


def _optimizer_mismatch_claim_contradicts_evidence(state: RState, graph_text: str) -> bool:
    passed_optimizer = _extract_passed_optimizer_from_args(state, graph_text)
    supported = _extract_supported_optimizers_from_code(state)
    if not passed_optimizer or not supported:
        return False
    return passed_optimizer in supported


def _looks_like_data_shape_mismatch_failure(outcome: str, reason: str | None) -> bool:
    text = (outcome + " " + (reason or "")).lower()
    shape_terms = ["mismatch", "shape", "dimension", "size", "format", "28x28", "32x32", "32 x 32", "28 x 28"]
    failure_terms = ["fail", "error", "cannot run", "won't run", "crash"]
    return any(term in text for term in shape_terms) and any(term in text for term in failure_terms)


def _has_explicit_unresolved_shape_mismatch_evidence(state: RState, graph_text: str) -> bool:
    evidence_text = (get_state_code(state) + "\n" + str(state.args) + "\n" + graph_text).lower()
    explicit_error_terms = [
        "size mismatch",
        "shapes cannot be multiplied",
        "expected input",
        "mat1 and mat2",
        "runtimeerror",
        "dimension out of range",
    ]
    has_explicit_error = any(term in evidence_text for term in explicit_error_terms)
    has_resize_to_32 = "resize((32, 32))" in evidence_text or "resize((32,32))" in evidence_text
    mentions_mnist = "mnist" in evidence_text
    if has_resize_to_32 and mentions_mnist and not has_explicit_error:
        return False
    return has_explicit_error


def convert_graph_to_text(graph: ComputeGraph, state: RState) -> str:
    """
    Convert a graph  representation to a text format for LLM input.
    This is a placeholder implementation; you can customize it as needed.
    """
    # get transformations that are connected to the given state
    transformations = graph.query_transformations(to_state=state.name)
    graph_text = ""
    for transformation in transformations:
        graph_text += f"  State: {transformation.state1.name} [{transformation.state1.status}]\n"
        graph_text += f"  Args Diff vs Current: {transformation.args_changes_str()}\n"
        graph_text += f"  Code Changes: {transformation.p_changes}\n"
        graph_text += f"  Context Changes: {format_context_changes_for_llm(transformation.context_changes)}\n\n"
    return graph_text

def convert_graph_to_text_states(graph: ComputeGraph, state: RState, outcomes:dict[str, Outcome]) -> str:
    """
    Convert a graph representation to a text format for LLM input.
    This is a placeholder implementation; you can customize it as needed.
    """
    # get states that are similar to the given state
    transformations = graph.query_transformations(to_state=state.name)
    graph_text = ""
    for transformation in transformations:
        graph_text += f"  State: {transformation.state1.name} [{transformation.state1.status}]\n"
        graph_text += f"  Args Diff vs Current: {transformation.args_changes_str()}\n"
        graph_text += f"  Code Changes: {transformation.p_changes}\n"
        graph_text += f"  Context Changes: {format_context_changes_for_llm(transformation.context_changes)}\n"
        outcome = outcomes.get(transformation.state1.name)
        if outcome:
            graph_text += f"  Outcome: {outcome.outputs}\n"
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
    def _set_graph_builder(self, graph_builder: GraphBuilder):
        self.graph_builder = graph_builder

    def predict_outcome(self, state: RState) -> OutcomeLabel:
        with (module_path / "prompts" / "outcome_prediction.txt").open("r") as f:
            prompt_template = f.read()
        syntax_ok, syntax_details = check_python_syntax(state.procedure_source)
        syntax_evidence = f"procedure_source_python_syntax={syntax_details}"
        graph_text = self.convert_graph_to_text_states(state)
        prompt = prompt_template.format(
            graph = graph_text,
            code = get_state_code(state),
            args = state.args,
            syntax_evidence = syntax_evidence,
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
            unsupported_failure_claim = False
            unsupported_reason = None

            if _looks_like_hard_failure(outcome_label, reason):
                if syntax_ok and _looks_like_syntax_failure(outcome_label, reason):
                    unsupported_failure_claim = True
                    unsupported_reason = (
                        "Syntax-related failure is unsupported because deterministic parsing of "
                        "procedure_source succeeded."
                    )
                elif _looks_like_optimizer_name_mismatch_failure(outcome_label, reason) and _optimizer_mismatch_claim_contradicts_evidence(state, graph_text):
                    unsupported_failure_claim = True
                    unsupported_reason = (
                        "Optimizer-name mismatch failure is unsupported because the passed optimizer appears "
                        "in supported code branches."
                    )
                elif _looks_like_unsupported_optimizer_failure(outcome_label, reason) and not _has_explicit_optimizer_claim_evidence(state, graph_text, outcome_label, reason):
                    unsupported_failure_claim = True
                    unsupported_reason = (
                        "Unsupported-optimizer failure is ungrounded because the claimed optimizer value "
                        "is not explicitly present in provided args/history."
                    )
                elif _looks_like_data_shape_mismatch_failure(outcome_label, reason) and not _has_explicit_unresolved_shape_mismatch_evidence(state, graph_text):
                    unsupported_failure_claim = True
                    unsupported_reason = (
                        "Data-shape mismatch failure is unsupported because no explicit unresolved mismatch "
                        "evidence is present in provided inputs."
                    )
                elif _looks_like_env_device_failure(outcome_label, reason) and not _has_explicit_env_device_evidence(state, graph_text):
                    unsupported_failure_claim = True
                    unsupported_reason = (
                        "Environment/device failure is unsupported because required device evidence is missing "
                        "from provided code/args/history."
                    )

            if unsupported_failure_claim:
                outcome_label = "likely non-fatal outcome; any failure risk is unconfirmed from provided evidence"
                confidence = min(confidence, 0.45)
                reason = unsupported_reason
                confidence_reason = (
                    "Confidence reduced because the predicted hard-failure mode is not grounded in deterministic "
                    "or explicit input evidence."
                )
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
    
    def convert_graph_to_text_states(self, state: RState) -> str:
        """
        Convert a graph representation to a text format for LLM input.
        This is a placeholder implementation; you can customize it as needed.
        """
        # get states that are similar to the given state
        transformations = self.graph_builder.compute_graph.query_transformations(to_state=state.name)
        graph_text = ""
        for transformation in transformations:
            graph_text += f"  State: {transformation.state1.name} [{transformation.state1.status}]\n"
            graph_text += f"  Args Diff vs Current: {transformation.args_changes_str()}\n"
            graph_text += f"  Code Changes: {transformation.p_changes}\n"
            graph_text += f"  Context Changes: {format_context_changes_for_llm(transformation.context_changes)}\n"
            outcome = self.graph_builder.load_outcome(transformation.state1)
            if outcome:
                graph_text += f"  Outcome: {outcome.outputs}\n"
        return graph_text