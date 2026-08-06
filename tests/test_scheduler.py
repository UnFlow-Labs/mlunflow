from unittest.mock import MagicMock

import pytest

from unflow.core.executors.local_exectuor import LocalExecutor
from unflow.core.scheduler import ExecutionRecord, Scheduler
from unflow.core.unflow_types import Outcome, RState, RStateStatus
from unflow.graph.compute_graph import ComputeGraph


def _id(x):
    return x


def make_state(name, func=_id, args=None):
    return RState(name=name, procedure=func, args=args or {"x": 1})


class TestExecutionRecord:
    def test_default_values(self):
        r = ExecutionRecord()
        assert r.status == RStateStatus.PENDING
        assert r.outcome is None
        assert r.error is None


class TestScheduler:
    @pytest.fixture
    def single_graph(self):
        g = ComputeGraph("unflow_scheduler")
        g.add_state(make_state("s1"))
        return g

    @pytest.fixture
    def linear_graph(self):
        g = ComputeGraph("unflow_scheduler")
        s1 = make_state("s1")
        s2 = make_state("s2")
        s3 = make_state("s3")
        g.add_state(s1)
        g.add_state(s2)
        g.add_state(s3)
        g.add_transformation(s1, s2, MagicMock())
        g.add_transformation(s2, s3, MagicMock())
        return g

    def test_init__creates_records(self, single_graph):
        sched = Scheduler(single_graph, LocalExecutor())
        assert "s1" in sched.records
        assert sched.records["s1"].status == RStateStatus.PENDING

    def test_ready__no_predecessors(self, single_graph):
        sched = Scheduler(single_graph, LocalExecutor())
        node = next(iter(single_graph.graph.nodes()))
        assert sched._ready(node) is True

    def test_ready__all_predecessors_completed(self, linear_graph):
        sched = Scheduler(linear_graph, LocalExecutor())
        s1 = "s1"
        s2 = "s2"
        sched.records[s1].status = RStateStatus.COMPLETED
        assert sched._ready(s2) is True

    def test_ready__predecessor_not_completed(self, linear_graph):
        sched = Scheduler(linear_graph, LocalExecutor())
        s2 = "s2"
        assert sched._ready(s2) is False

    def test_run__single_node(self, single_graph):
        sched = Scheduler(single_graph, LocalExecutor())
        target = single_graph.get_state("s1")
        record, _ = sched.run(target)
        assert record.status == RStateStatus.COMPLETED
        assert record.outcome is not None
        assert record.outcome.outputs == 1

    def test_run__linear_chain(self, linear_graph):
        sched = Scheduler(linear_graph, LocalExecutor())
        target = linear_graph.get_state("s3")
        record, _ = sched.run(target)
        assert record.status == RStateStatus.COMPLETED
        for name in ["s1", "s2", "s3"]:
            assert sched.records[name].status == RStateStatus.COMPLETED

    def test_run__already_completed_node_skipped(self, single_graph):
        sched = Scheduler(single_graph, LocalExecutor())
        target = single_graph.get_state("s1")
        sched.records["s1"].status = RStateStatus.COMPLETED
        sched.records["s1"].outcome = Outcome(target, 99)
        record, _ = sched.run(target)
        assert record.outcome is not None
        assert record.outcome.outputs == 99
        assert record.status == RStateStatus.COMPLETED

    def test_run__returns_graph(self, single_graph):
        sched = Scheduler(single_graph, LocalExecutor())
        target = single_graph.get_state("s1")
        _, graph = sched.run(target)
        assert graph is not None


class TestSchedulerParallel:
    """Tests for parallel execution support in the scheduler."""

    def make_diamond_graph(self):
        """
        Creates a diamond DAG:
            s1
           /  \
          s2  s3
           \  /
            s4
        """
        g = ComputeGraph("unflow_scheduler")
        s1 = make_state("s1")
        s2 = make_state("s2")
        s3 = make_state("s3")
        s4 = make_state("s4")
        g.add_state(s1)
        g.add_state(s2)
        g.add_state(s3)
        g.add_state(s4)
        g.add_transformation(s1, s2, MagicMock())
        g.add_transformation(s1, s3, MagicMock())
        g.add_transformation(s2, s4, MagicMock())
        g.add_transformation(s3, s4, MagicMock())
        return g

    def test_run__diamond_all_completed(self):
        g = self.make_diamond_graph()
        sched = Scheduler(g, LocalExecutor())
        target = g.get_state("s4")
        record, _ = sched.run(target)
        assert record.status == RStateStatus.COMPLETED
        for name in ["s1", "s2", "s3", "s4"]:
            assert sched.records[name].status == RStateStatus.COMPLETED

    # def test_run__diamond_partial_failure(self):
    #     """If one branch fails, the dependent node should not run."""
    #     g = self.make_diamond_graph()
    #     g.get_state("s3").procedure = lambda x: (_ for _ in ()).throw(RuntimeError("boom"))
    #     sched = Scheduler(g, LocalExecutor())
    #     target = g.get_state("s4")

    #     _, _ = sched.run(target)
    #     assert sched.records["s3"].status == RStateStatus.FAILED
    #     assert sched.records["s4"].status == RStateStatus.PENDING

    def test_run__two_independent_chains(self):
        """Two independent chains: s1->s2 and s3->s4. Only the target's chain runs."""
        g = ComputeGraph("unflow_scheduler")
        s1 = make_state("s1")
        s2 = make_state("s2")
        s3 = make_state("s3")
        s4 = make_state("s4")
        g.add_state(s1)
        g.add_state(s2)
        g.add_state(s3)
        g.add_state(s4)
        g.add_transformation(s1, s2, MagicMock())
        g.add_transformation(s3, s4, MagicMock())

        sched = Scheduler(g, LocalExecutor())

        # Run s2 — only s1 and s2 should execute
        record, _ = sched.run(g.get_state("s2"))
        assert record.status == RStateStatus.COMPLETED
        assert sched.records["s1"].status == RStateStatus.COMPLETED
        assert sched.records["s2"].status == RStateStatus.COMPLETED
        # s3 and s4 should still be PENDING (not in required subgraph)
        assert sched.records["s3"].status == RStateStatus.PENDING
        assert sched.records["s4"].status == RStateStatus.PENDING
