from __future__ import annotations

import networkx as nx

from unflow.core.executors.executor import Executor
from unflow.core.unflow_types import ExecutionRecord, Job, Outcome, RState, RStateStatus
from unflow.graph.compute_graph import ComputeGraph


class Scheduler:
    def __init__(self, graph: ComputeGraph, executor: Executor):
        self.graph = graph.graph
        self.executor = executor

        self.records = {node.name: ExecutionRecord(node.status, state_name=node.name) for node in graph.get_states()}

    def _ready(self, node) -> bool:
        """Returns True if all dependencies completed."""
        return all(
            self.records[pred].status == RStateStatus.COMPLETED or self.records[pred].status == RStateStatus.RUNNING
            for pred in self.graph.predecessors(node)
        )

    def run(self, target: RState):

        # ----------------------------------------
        # Build the required subgraph
        # ----------------------------------------
        target = self.graph.nodes[target.name]["state"]
        ancestors = nx.ancestors(self.graph, target.name)
        required = ancestors | {target.name}

        subgraph = self.graph.subgraph(required)

        # ----------------------------------------
        # Track outstanding jobs
        # ----------------------------------------

        outstanding: dict[str, Job] = {}

        def _submit(node: str):
            state = self.graph.nodes[node]["state"]
            self.records[node].status = RStateStatus.READY
            self.records[node].status = RStateStatus.RUNNING
            job = self.executor.submit(state)
            outstanding[node] = job

        # ----------------------------------------
        # Submit all initially ready nodes
        # ----------------------------------------

        for node in nx.topological_sort(subgraph):
            if self.records[node].status == RStateStatus.COMPLETED or self.records[node].status == RStateStatus.FAILED:
                continue
            if self._ready(node):
                _submit(node)
                print(f"Submitted {node} for execution.")
            else:
                print(f"{node} is not ready for execution. Waiting for dependencies.")

        # ----------------------------------------
        # Main scheduling loop
        # ----------------------------------------

        while outstanding:
            job = self.executor.wait()
            if job is None:
                break

            # Find which node this job belongs to
            node_name = next(n for n, j in outstanding.items() if j is job)
            del outstanding[node_name]
            record = self.records[node_name]

            if job.error:
                record.error = job.error
                record.status = RStateStatus.FAILED
                self.graph.nodes[node_name]["state"].status = RStateStatus.FAILED
                print(f"{node_name} failed: {job.error}")
            else:
                record.status = RStateStatus.COMPLETED
                self.graph.nodes[node_name]["state"].status = RStateStatus.COMPLETED
            record.outcome = Outcome(self.graph.nodes[node_name]["state"], job.output)
            record.start_time = job.start_time
            record.end_time = job.end_time

            # ----------------------------------------
            # Check successors
            # ----------------------------------------

            for succ in self.graph.successors(node_name):
                if succ not in required:
                    continue

                succ_record = self.records[succ]

                if succ_record.status == RStateStatus.PENDING and self._ready(succ):
                    _submit(succ)
        return self.records[target.name], self.graph
