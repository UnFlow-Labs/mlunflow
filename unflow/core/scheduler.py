from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from unflow.core.executors.executor import Executor
from unflow.core.unflow_types import RState, RStateStatus
from unflow.graph.compute_graph import ComputeGraph


@dataclass
class ExecutionRecord:
    status: RStateStatus = RStateStatus.PENDING
    outputs: object = None
    error: Exception | None = None


class Scheduler:
    def __init__(self, graph: ComputeGraph, executor: Executor):
        self.graph = graph.graph
        self.executor = executor

        self.records = {node.name: ExecutionRecord(node.status) for node in graph.get_states()}

    def _ready(self, node) -> bool:
        """Returns True if all dependencies completed."""
        return all(self.records[pred].status == RStateStatus.COMPLETED for pred in self.graph.predecessors(node))

    def run(self, target: RState):

        # ----------------------------------------
        # Build the required subgraph
        # ----------------------------------------
        target = self.graph.nodes[target.name]["state"]
        ancestors = nx.ancestors(self.graph, target.name)
        required = ancestors | {target.name}

        subgraph = self.graph.subgraph(required)

        # ----------------------------------------
        # Initial ready queue
        # ----------------------------------------

        queue: deque[str] = deque()

        for node in nx.topological_sort(subgraph):
            if self.records[node].status == RStateStatus.COMPLETED:
                continue
            if self._ready(node):
                self.records[node].status = RStateStatus.READY
                queue.append(node)

        # ----------------------------------------
        # Main scheduling loop
        # ----------------------------------------

        while queue:
            node = queue.popleft()

            record = self.records[node]

            if record.status == RStateStatus.COMPLETED:
                continue

            state = self.graph.nodes[node]["state"]

            try:
                record.status = RStateStatus.RUNNING

                job = self.executor.submit(state)
                while not job.completed:
                    job = self.executor.wait()
                if job.error:
                    raise job.error
                self.graph.nodes[node]["state"].status = RStateStatus.COMPLETED
                record.outputs = job.output
                record.status = RStateStatus.COMPLETED

            except Exception as e:
                record.error = e
                record.status = RStateStatus.FAILED
                self.graph.nodes[node]["state"].status = RStateStatus.FAILED
                print(f"{node} failed: {e}")
                continue

            # ----------------------------------------
            # Check successors
            # ----------------------------------------

            for succ in self.graph.successors(node):
                if succ not in required:
                    continue

                succ_record = self.records[succ]

                if succ_record.status == RStateStatus.PENDING and self._ready(succ):
                    succ_record.status = RStateStatus.READY
                    queue.append(succ)

        return self.records[target.name], self.graph
