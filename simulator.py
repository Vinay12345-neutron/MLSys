import json
import math
from typing import List, Dict, Optional, Tuple, Any

class Tensor:
    def __init__(self, id: int, width: int, height: int):
        self.id = id
        self.width = width
        self.height = height
        self.size = width * height

class Op:
    def __init__(self, id: int, op_type: str, inputs: List[int], outputs: List[int], base_cost: float):
        self.id = id
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.base_cost = base_cost

class Problem:
    def __init__(self, data: Dict[str, Any]):
        self.tensors = [Tensor(i, w, h) for i, (w, h) in enumerate(zip(data["widths"], data["heights"]))]
        self.ops = [Op(i, t, ins, outs, c) for i, (t, ins, outs, c) in enumerate(zip(data["op_types"], data["inputs"], data["outputs"], data["base_costs"]))]
        self.fast_memory_capacity = data["fast_memory_capacity"]
        self.slow_memory_bandwidth = data["slow_memory_bandwidth"]
        self.native_granularity = data["native_granularity"]

class Granularity:
    def __init__(self, width: int, height: int, depth: int):
        self.width = width
        self.height = height
        self.depth = depth

class Subgraph:
    def __init__(self, ops: List[int], tensors_to_retain: List[int], granularity: List[int], traversal_order: Optional[List[int]], subgraph_latency: float):
        self.ops = ops
        self.tensors_to_retain = tensors_to_retain
        self.granularity = Granularity(*granularity)
        self.traversal_order = traversal_order
        self.subgraph_latency = subgraph_latency

class Solution:
    def __init__(self, data: Dict[str, Any]):
        self.subgraphs = []
        for i in range(len(data.get("subgraphs", []))):
            self.subgraphs.append(Subgraph(
                ops=data["subgraphs"][i],
                tensors_to_retain=data["tensors_to_retain"][i],
                granularity=data["granularities"][i],
                traversal_order=data["traversal_orders"][i] if "traversal_orders" in data and i < len(data["traversal_orders"]) else None,
                subgraph_latency=data.get("subgraph_latencies", [0]*len(data.get("subgraphs", [])))[i]
            ))

def evaluate(problem: Problem, solution: Solution) -> Tuple[bool, str, float]:
    total_latency = 0.0
    executed_ops = set()
    resident_tensors = set() # Tracks what is currently sitting on the chip

    for step_idx, sub in enumerate(solution.subgraphs):
        w, h, k = sub.granularity.width, sub.granularity.height, sub.granularity.depth
        
        # --- 1. THE OOM CHECK (Strict Memory Bounds) ---
        working_set_bytes = 0
        produced_in_subgraph = set()
        required_tensors = set()
        
        for op_id in sub.ops:
            op = problem.ops[op_id]
            produced_in_subgraph.update(op.outputs)
            for in_id in op.inputs:
                if in_id not in produced_in_subgraph:
                    required_tensors.add(in_id)

        # Calculate Tile Footprint (The memory needed just for the active math)
        for t_id in required_tensors:
            working_set_bytes += (w * k) # Approximating input tile size
        for t_id in produced_in_subgraph:
            working_set_bytes += (w * h) # Approximating output tile size
            
        # Add full tensors deliberately kept on the chip for the next step
        for t_id in sub.tensors_to_retain:
            working_set_bytes += problem.tensors[t_id].size

        # THE CRITICAL CHECK
        if working_set_bytes > problem.fast_memory_capacity:
            return False, f"OOM in subgraph {step_idx}: Used {working_set_bytes} bytes, limit is {problem.fast_memory_capacity}.", 0.0


        # --- 2. THE ROOFLINE MODEL (True Latency Calculation) ---
        
        # Assuming the tensors being processed are 128x128 based on your problem JSON
        tensor_w, tensor_h = 128, 128 
        
        # How many tiles do we need to cover the full tensor?
        # math.ceil ensures we count partial tiles at the edges
        tiles_needed = math.ceil(tensor_w / w) * math.ceil(tensor_h / h)

        # A. Compute Time (How long the math takes for ALL tiles)
        compute_cost = 0
        for op_id in sub.ops:
            op = problem.ops[op_id]
            if op.op_type == "MatMul":
                compute_cost += (w * h * k) * op.base_cost * tiles_needed
            else:
                compute_cost += (w * h) * op.base_cost * tiles_needed

        # B. Memory Transfer Time (How long the data takes to arrive for ALL tiles)
        bytes_transferred = 0
        for t_id in required_tensors:
            if t_id not in resident_tensors: # Only pay cost if it's not already on chip!
                bytes_transferred += problem.tensors[t_id].size * tiles_needed
        for t_id in produced_in_subgraph:
            if t_id not in sub.tensors_to_retain: # Only pay cost if it's being evicted
                bytes_transferred += problem.tensors[t_id].size * tiles_needed
                
        memory_time = bytes_transferred / problem.slow_memory_bandwidth
        
        # Hardware bottleneck: A step takes as long as its slowest pipeline stage
        step_latency = max(compute_cost, memory_time)
        total_latency += step_latency

        # --- 3. STATE UPDATE ---
        resident_tensors = set(sub.tensors_to_retain)
        for op_id in sub.ops:
            if op_id in executed_ops:
                return False, f"Topology Error: Op {op_id} executed multiple times.", 0.0
            executed_ops.add(op_id)

    if len(executed_ops) != len(problem.ops):
        return False, f"Topology Error: Executed {len(executed_ops)} out of {len(problem.ops)} ops.", 0.0

    return True, "Success", total_latency

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        try:
            with open(sys.argv[1]) as f:
                prob_data = json.load(f)
            with open(sys.argv[2]) as f:
                sol_data = json.load(f)
            prob = Problem(prob_data)
            sol = Solution(sol_data)
            valid, msg, lat = evaluate(prob, sol)
            print(f"Valid: {valid}, Message: {msg}, Latency: {lat}")
        except Exception as e:
            print(f"Valid: False, Message: Error parsing - {e}, Latency: 0.0")
