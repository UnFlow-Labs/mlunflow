import json
import sys
from pathlib import Path

# Allow running as a plain script: python examples/nanogpt/build_graph.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from unflow.core.builder import GraphBuilder
from unflow.core.unflow_types import Outcome, RState, RStateStatus
from unflow.graph.compute_graph import ComputeGraph
from unflow.reasoning.detection import LLMIntentDetector, OutcomeLabel
from unflow.reasoning.llm import LlamaLLM, TransformersLLM, OllamaLLM, LLMConfig
from unflow.core.unflow_core import unflowdecorator

EXAMPLES_DIR = Path(__file__).resolve().parent
SPEEDRUN_DIR = EXAMPLES_DIR / "nanogpt_speedrun"
GRAPH_NAME = "nanogpt_speedrun"

execution_path = str(EXAMPLES_DIR)
builder = GraphBuilder(execution_path)
builder.load_graph(GRAPH_NAME)
intent_detector = LLMIntentDetector(llm=OllamaLLM(model_name="nemotron-3-ultra:cloud"))
intent_detector._set_graph(builder.compute_graph)
intent_detector._set_graph_builder(builder)

def extract_training_time_from_outcome(outcome_label: OutcomeLabel) -> float | None:
    '''Extract the training time from the predicted outcome.'''
    import re

    text = outcome_label.outcome

    # Pattern 1: Explicit "train_time" or "training time" with a number
    # Matches: "train_time around 770,000 ms", "training time ~30-60 minutes",
    #          "total training time between 1,000,000 and 1,100,000 ms"
    pattern = re.IGNORECASE
    time_patterns = [
        # train_time/training time followed by a number (with optional commas/k suffix)
        r"(?:total\s+)?train(?:ing)?[_\s]?time\s*(?:of|is|around|~|approximately|about|between|will be|expected)?\s*~?\s*(\d[\d,]*\.?\d*)\s*(k|m)?\s*(ms|milliseconds?|minutes?|mins?|hours?|hrs?)?",
        # "between X and Y ms/minutes" pattern
        r"between\s+(\d[\d,]*\.?\d*)\s*(k|m)?\s*(ms|milliseconds?|minutes?|mins?|hours?|hrs?)?\s+and\s+(\d[\d,]*\.?\d*)\s*(k|m)?\s*(ms|milliseconds?|minutes?|mins?|hours?|hrs?)?",
        # "X-Y ms/minutes" range pattern
        r"(\d[\d,]*\.?\d*)\s*(k|m)?\s*(ms|milliseconds?|minutes?|mins?|hours?|hrs?)?\s*[-–]\s*(\d[\d,]*\.?\d*)\s*(k|m)?\s*(ms|milliseconds?|minutes?|mins?|hours?|hrs?)?",
    ]

    for pat in time_patterns:
        match = re.search(pat, text, pattern)
        if match:
            groups = match.groups()
            # Handle "between X and Y" pattern (6 groups)
            if len(groups) >= 6 and groups[0] and groups[3]:
                val1 = _parse_number(groups[0], groups[1])
                val2 = _parse_number(groups[3], groups[4])
                unit = groups[2] or groups[5] or "ms"
                return _convert_to_ms((val1 + val2) / 2, unit)
            # Handle "X-Y" range pattern (6 groups)
            elif len(groups) >= 6 and groups[0] and groups[3]:
                val1 = _parse_number(groups[0], groups[1])
                val2 = _parse_number(groups[3], groups[4])
                unit = groups[2] or groups[5] or "ms"
                return _convert_to_ms((val1 + val2) / 2, unit)
            # Handle single value pattern (3 groups)
            elif groups[0]:
                val = _parse_number(groups[0], groups[1])
                unit = groups[2] or "ms"
                return _convert_to_ms(val, unit)

    return None


def _parse_number(num_str: str, suffix: str | None) -> float:
    """Parse a number string that may contain commas and a k/m suffix."""
    num_str = num_str.replace(",", "")
    val = float(num_str)
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val


def _convert_to_ms(val: float, unit: str) -> float:
    """Convert a value from the given unit to milliseconds."""
    unit = unit.lower()
    if unit in ("ms", "millisecond", "milliseconds"):
        return val
    elif unit in ("min", "mins", "minute", "minutes"):
        return val * 60_000
    elif unit in ("hr", "hrs", "hour", "hours"):
        return val * 3_600_000
    return val


print("Outcomes (from results.json):")
for i, state in enumerate(sorted(builder.compute_graph.get_states(), key=lambda s: s.args.get("record_id", 0))):
    if i >= 15: continue
    print("state:", state.name, "status:", state.status.value
          )
    outcome_label = intent_detector.predict_outcome(state)
    print(f"  {state.name}: status={state.status.value}, predicted outcome: {outcome_label}")
    with open(f"{state.name}_predicted_outcome_no_outcome.json", "w") as f:
        json.dump(outcome_label.__dict__(), f, indent=2)
    with open(f"{state.name}_predicted_outcome_no_outcome.json", "r") as f:
        predicted_outcome = json.load(f)
    predicted_outcome = OutcomeLabel(
        outcome=predicted_outcome["label"],
        confidence=predicted_outcome["confidence"],
        metadata=predicted_outcome.get("metadata", {}),
        reason=predicted_outcome.get("reason", ""),
        confidence_reason=predicted_outcome.get("confidence_reason", ""),
    )
    predicted_training_time = extract_training_time_from_outcome(predicted_outcome)
    if predicted_training_time is not None:
        # convert to minutes if it's in milliseconds
        if predicted_training_time > 1000:
            predicted_training_time = predicted_training_time / 60000  # convert ms to minutes

    outcome = builder.load_outcome(state)
    val_loss = None
    time = outcome.outputs["train_time"] if outcome.outputs else None
    # convert time to minutes if it's in milliseconds
    if time is not None and time > 1000:
        time = time / 60000  # convert ms to minutes
    print(f"  {state.name}: status={state.status.value}, possible predicted training time (minutes): {predicted_training_time}, time={time}")

