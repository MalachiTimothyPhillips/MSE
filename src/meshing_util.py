import numpy as np
class MeshNode:
    def __init__(self, global_id, local_id, on_boundary):
        self.global_id = global_id
        self.local_id = local_id
        self.on_boundary = on_boundary
class Element:
    def __init__(self, p, nodes):
        self.nodes = nodes
class Mesh:
    def __init__(self, p):
        # hard code in E=2


