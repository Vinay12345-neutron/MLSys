import json


class Problem:
    def __init__(self, data):
        self.widths = data["widths"]
        self.heights = data["heights"]
        self.inputs = data["inputs"]
        self.outputs = data["outputs"]
        self.base_costs = data["base_costs"]
        self.op_types = data["op_types"]
        self.fast_memory_capacity = data["fast_memory_capacity"]
        self.slow_memory_bandwidth = data["slow_memory_bandwidth"]
        self.native_granularity = data["native_granularity"]

    def get_tensor_size(self, idx):
        return self.widths[idx] * self.heights[idx]


class Solution:
    def __init__(self, data):
        self.subgraphs = data["subgraphs"]
        self.granularities = data["granularities"]
        self.tensors_to_retain = data["tensors_to_retain"]
        self.traversal_orders = data.get(
            "traversal_orders", [None] * len(data["subgraphs"])
        )


def evaluate(problem: Problem, solution: Solution):
    all_produced_tensors = set()
    for op_outputs in problem.outputs:
        all_produced_tensors.update(op_outputs)

    all_tensors = set(range(len(problem.widths)))

    ready_tensors = all_tensors - all_produced_tensors

    ran_ops = set()
    for sg_idx, ops_in_sg in enumerate(solution.subgraphs):
        for op_idx in ops_in_sg:
            # Check A: Has this op already been run? (No duplicates allowed)
            # if op_idx in scheduled_ops:
            #     return [0, f"Error: Op[{op_idx}] scheduled multiple times.", [], 0.0]

            # Check B: Are the inputs for this Op ready?
            required_inputs = problem.inputs[op_idx]
            for t_in in required_inputs:
                if t_in not in ready_tensors:
                    return [
                        0,
                        f"Error: Op[{op_idx}] in Subgraph {sg_idx} requires Tensor[{t_in}], but it hasn't been produced yet.",
                        [],
                        0.0,
                    ]

            # Check C: "Execute" the op by marking its outputs as ready
            produced_outputs = problem.outputs[op_idx]
            ready_tensors.update(produced_outputs)
            ran_ops.add(op_idx)

    total_ops_in_problem = set(range(len(problem.base_costs)))
    if not total_ops_in_problem.issubset(ran_ops):
        missing = total_ops_in_problem - ran_ops
        return [
            0,
            f"Error: Ops {missing} were never executed in any subgraph.",
            [],
            0.0,
        ]

    resident_tensors = set()

    for sg_idx, ops_in_sg in enumerate(solution.subgraphs):
        gw, gh, gk = solution.granularities[sg_idx]
        if gw <= 0 or gh <= 0 or gk <= 0:
            return [
                0,
                f"Error: Subgraph {sg_idx} has invalid granularity dimensions: {[gw, gh, gk]}, the granularities cant be 0 or negative",
                [],
                0.0,
            ]

        ref_t_idx = problem.outputs[ops_in_sg[0]][0]
        full_w = problem.widths[ref_t_idx]
        full_h = problem.heights[ref_t_idx]

        if gw > full_w or gh > full_h:
            return [
                0,
                f"Error: Subgraph {sg_idx} granularity {gw}x{gh} exceeds tensor dimensions {full_w}x{full_h}",
                [],
                0.0,
            ]
        native_w, native_h = problem.native_granularity

        tensor_slice_shapes = {}
        last_op = ops_in_sg[-1]

        for out_t in problem.outputs[last_op]:
            tensor_slice_shapes[out_t] = (gh, gw)

    return 0


# # --- Example Usage ---
if __name__ == "__main__":
    # Mocking Example 1 Strategy B
    prob = Problem(
        {
            "widths": [128, 128, 128],
            "heights": [128, 128, 128],
            "inputs": [[0], [1]],
            "outputs": [[1], [2]],
            "base_costs": [1000, 100],
            "op_types": ["Pointwise", "Pointwise"],
            "fast_memory_capacity": 35000,
            "slow_memory_bandwidth": 10,
            "native_granularity": [128, 128],
        }
    )

    sol = Solution(
        {
            "subgraphs": [[0, 1]],
            "granularities": [[128, 128, 1]],
            "tensors_to_retain": [[]],
        }
    )

    result = evaluate(prob, sol)
    print(f"Valid: {result[0]}")
    print(f"Message: '{result[1]}'")
    print(f"Subgraph Latencies: {result[2]}")
    print(f"Total Latency: {result[3]}")
