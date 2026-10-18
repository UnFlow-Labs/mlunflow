"""Build an unflow compute graph from the nanogpt_speedrun records.

Each directory under ``nanogpt_speedrun/`` (``record_1`` ... ``record_21``)
holds one completed GPT-2 speedrun experiment:

- ``train_gpt2.py``: the training script used for that run (the procedure)
- ``results.json``:   the outcome of that run (metrics, hypothesis, summary)

The training itself is never re-executed. Every record is reconstructed as
a ``COMPLETED`` state whose procedure source is the record's training
script, and the content of its ``results.json`` is stored as the state's
outcome in the unflow database.
"""

import json
import sys
from pathlib import Path

# Allow running as a plain script: python examples/nanogpt/build_graph.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


from unflow.core.builder import GraphBuilder
from unflow.core.unflow_types import Outcome, RState, RStateStatus, Procedure
from unflow.graph.compute_graph import ComputeGraph
from unflow.reasoning.detection import LLMIntentDetector
from unflow.reasoning.llm import LlamaLLM
from unflow.core.unflow_core import unflowdecorator

EXAMPLES_DIR = Path(__file__).resolve().parent
SPEEDRUN_DIR = EXAMPLES_DIR / "nanogpt_speedrun"
GRAPH_NAME = "nanogpt_speedrun"


def _placeholder_procedure():
    """Stand-in procedure for historical runs; training is never re-executed."""
    return None


def load_records(speedrun_dir: Path) -> list[dict]:
    """Load every ``record_<id>`` directory under ``speedrun_dir``, sorted by id."""
    records = []
    for record_dir in sorted(speedrun_dir.glob("record_*"), key=lambda p: int(p.name.split("_")[1])):
        script_path = record_dir / "train_gpt2.py"
        results_path = record_dir / "results.json"
        if not script_path.is_file() or not results_path.is_file():
            print(f"Skipping {record_dir.name}: missing train_gpt2.py or results.json")
            continue
        results = json.loads(results_path.read_text())
        records.append(
            {
                "id": int(record_dir.name.split("_")[1]),
                "name": record_dir.name,
                "script_source": script_path.read_text(),
                "results": results,
                "script_path": str(script_path),
            }
        )
    return records


def build_graph(
    speedrun_dir: Path = SPEEDRUN_DIR,
    execution_path: str | None = None,
) -> ComputeGraph:
    """Reconstruct the unflow compute graph from the speedrun records.

    Each record becomes one ``COMPLETED`` state (the training is not re-run)
    and its ``results.json`` is stored as the state's outcome. The graph and
    the outcomes are persisted to the unflow database under ``GRAPH_NAME`` so
    they can be reloaded with ``GraphBuilder.load_graph``/``load_outcome``.
    """
    execution_path = execution_path or str(EXAMPLES_DIR)
    builder = GraphBuilder(execution_path)
    # intent_detector = LLMIntentDetector(llm=LlamaLLM(repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", 
    #     filename="qwen2.5-coder-7b-instruct-q8_0.gguf", n_ctx=4096))
    # intent_detector._set_graph(builder.compute_graph)
    # intent_detector._set_graph_builder(builder)
    added: list[tuple[RState, dict, str]] = []
    for i,record in enumerate(load_records(speedrun_dir)):
        state = RState(
            name=record["name"],
            procedure=Procedure(script_path=Path(record["script_path"])),
            args=None,
            description="",
        )
        can_transform, _ = builder.transform(state)
        if not can_transform:
            print(f"Skipping {record['name']}: identical to an existing state")
            continue
        # The run already happened: mark the node as complete.
        # intent = intent_detector.detect_intent(state)
        # print(f"Record {record['name']} intent: {intent}")
        # state.infered_intent = intent
        # builder.save_graph(GRAPH_NAME)
        # if  i % 5 == 0:
        #     outcome_label = intent_detector.predict_outcome(state)
        #     print(f"Record {record['name']} predicted outcome: {outcome_label}")
        #     print(f"  Metrics: {record['results']['metrics']}")
        #     with open(f"{record['name']}_predicted_outcome.json", "w") as f:
        #         json.dump(outcome_label.__dict__(), f, indent=2)
        builder.set_status(state.name, RStateStatus.COMPLETED)
        added.append((state, record, ""))
    # Persist the graph first so the outcomes can be linked to it in the database.
    builder.save_graph(GRAPH_NAME)
    for state, record, _ in added:
        builder.save_outcome(state, Outcome(state, outputs={"train_time": record["results"]["metrics"]["train_time"],
                                                            "val_loss": record["results"]["metrics"]["val_loss"]}))
    return builder.compute_graph


def main() -> None:
    graph = build_graph()
    print(graph)
    print()

    # Verify the graph and outcomes round-trip through the database.
    check = GraphBuilder(str(EXAMPLES_DIR))
    check.load_graph(GRAPH_NAME)
    print(
        f"Reloaded from db: {check.compute_graph.graph.number_of_nodes()} states, "
        f"{check.compute_graph.graph.number_of_edges()} transformations"
    )
    print()
    print("Outcomes (from results.json):")
    for state in sorted(check.compute_graph.get_states(), key=lambda s: s.args.get("record_id", 0)):
        outcome = check.load_outcome(state)
        val_loss = outcome.outputs["val_loss"] if outcome.outputs else None
        print(outcome.outputs)
        time = outcome.outputs["train_time"] if outcome.outputs else None
        print(f"  {state.name}: status={state.status.value}, val_loss={val_loss}, time={time}")


if __name__ == "__main__":
    main()
