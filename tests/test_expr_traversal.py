from dataclasses import dataclass

import numpy as np

from rei import compile_nls_problem
from rei.core.expr import walk_expr
from rei.optimize.report import collect_named_expr_values


def test_reports_include_joint_power_inputs():
    runtime = compile_nls_problem({
        "variables": [{"name": "q", "dim": 2, "init": [1., 2.]}],
        "terms": [{"expr": {
            "type": "joint_power", "name": "power",
            "torque": {"type": "get_var", "name": "tau", "var": "q"},
            "velocity": {"type": "const", "name": "velocity", "var": "q", "value": [3., 4.]},
        }, "cost": {"type": "l2"}}],
    }, build_state=lambda *args, **kwargs: {})
    values = collect_named_expr_values(runtime)
    assert [item.name for item in values] == ["power", "tau", "velocity"]
    np.testing.assert_array_equal(values[0].value, [11.])
    np.testing.assert_array_equal(values[1].value, [1., 2.])


def test_explicit_children_support_custom_fields_shared_nodes_and_cycles():
    class Node:
        def __init__(self, *children):
            self.operands = children

        def children(self):
            return self.operands

    leaf = Node()
    left = Node(leaf)
    root = Node(left, leaf)
    leaf.operands = (root,)
    assert list(walk_expr(root)) == [root, left, leaf]


def test_legacy_custom_dataclass_composites_remain_traversable():
    @dataclass
    class Legacy:
        operands: object

        def eval(self, ctx):
            raise AssertionError("Traversal must not evaluate expressions")

    leaf = Legacy(None)
    root = Legacy({"nested": [leaf]})
    assert list(walk_expr(root)) == [root, leaf]
