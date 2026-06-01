"""
结果存储模块 — 树状结构记录特征之间的父子关系

用于追踪每个特征的推导路径，增强可解释性:
  原始特征 → 中间特征 → ... → 最终公式

汇报时可展示实际的推导树 → 证明算法"理解"了公式结构
"""


class FeatureNode:
    """特征树节点"""

    __slots__ = ('id', 'name', 'data', 'r2', 'loss', 'coef',
                 'parent_ids', 'operation', 'iteration')

    def __init__(self, node_id: int, name: str, data, r2: float = -1.0,
                 loss: float = 0.0, coef: list = None,
                 parent_ids: list = None, operation: str = '',
                 iteration: int = 0):
        self.id = node_id           # 唯一ID
        self.name = name            # 特征表达式字符串
        self.data = data            # 数值数据 (np array)
        self.r2 = r2                # R² 分数
        self.loss = loss            # MSE loss
        self.coef = coef or []      # 拟合系数 [c1, ..., b]
        self.parent_ids = parent_ids or []  # 父节点ID列表
        self.operation = operation  # 生成操作描述 (如 "a*b", "a/b")
        self.iteration = iteration  # 产生于第几轮迭代

    def get_full_name(self):
        """获取带系数的完整表达式"""
        if not self.coef:
            return self.name
        if len(self.coef) == 2:
            return f"{self.coef[0]:.4f} * {self.name} + {self.coef[1]:.4f}"
        return f"coef={self.coef} * {self.name}"

    def __repr__(self):
        short_name = self.name if len(self.name) <= 60 else self.name[:57] + '...'
        return (f"FeatureNode(id={self.id}, r2={self.r2:.4f}, name='{short_name}')")


class FeatureTree:
    """特征推导树 — 管理所有特征节点及其父子关系"""

    def __init__(self):
        self.nodes: dict[int, FeatureNode] = {}
        self._next_id = 0
        self.name_to_id: dict[str, int] = {}  # 表达式 -> 节点ID，用于去重
        self.best_node_id = None              # 当前最优特征节点ID

    def add_node(self, name: str, data, r2: float = -1.0,
                 loss: float = 0.0, coef: list = None,
                 parent_ids: list = None, operation: str = '',
                 iteration: int = 0) -> int:
        """添加节点，若表达式已存在且新R²更高则更新，否则跳过"""
        # 简化名称用于去重比较
        simplified = self._simplify_name(name)

        # 去重：已存在同名特征
        if simplified in self.name_to_id:
            existing_id = self.name_to_id[simplified]
            existing = self.nodes[existing_id]
            # 保留R²更高的版本
            if r2 > existing.r2:
                existing.r2 = r2
                existing.loss = loss
                existing.coef = coef or []
                existing.parent_ids = parent_ids or existing.parent_ids
                existing.operation = operation or existing.operation
                existing.iteration = iteration
            return existing_id

        node_id = self._next_id
        self._next_id += 1

        node = FeatureNode(
            node_id=node_id, name=name, data=data,
            r2=r2, loss=loss, coef=coef,
            parent_ids=parent_ids, operation=operation,
            iteration=iteration
        )
        self.nodes[node_id] = node
        self.name_to_id[simplified] = node_id

        # 更新最优
        if self.best_node_id is None or r2 > self.nodes[self.best_node_id].r2:
            self.best_node_id = node_id

        return node_id

    def add_original_features(self, data, feature_names: list):
        """添加原始特征作为根节点"""
        ids = []
        data_np = data.to_numpy()
        for i, name in enumerate(feature_names):
            nid = self.add_node(
                name=name, data=data_np[:, i],
                parent_ids=[], operation='原始特征', iteration=0
            )
            ids.append(nid)
        return ids

    def trace_path(self, node_id: int = None) -> list[FeatureNode]:
        """追溯从根到目标节点的推导路径"""
        if node_id is None:
            node_id = self.best_node_id
        if node_id is None:
            return []

        path = []
        visited = set()
        self._trace_recursive(node_id, path, visited)
        # 反转以得到从根到叶的顺序
        path.reverse()
        return path

    def _trace_recursive(self, node_id: int, path: list, visited: set):
        """递归追溯父节点"""
        if node_id in visited:
            return
        visited.add(node_id)
        node = self.nodes.get(node_id)
        if node is None:
            return

        path.append(node)
        for pid in node.parent_ids:
            self._trace_recursive(pid, path, visited)

    def get_best(self) -> FeatureNode:
        """获取当前最优特征"""
        if self.best_node_id is not None:
            return self.nodes[self.best_node_id]
        return None

    def get_derivation_tree_str(self, node_id: int = None) -> str:
        """生成推导树的字符串表示"""
        if node_id is None:
            node_id = self.best_node_id
        if node_id is None:
            return "(空)"

        lines = []
        self._print_tree(node_id, lines, indent=0, visited=set())
        return '\n'.join(lines)

    def _print_tree(self, node_id: int, lines: list, indent: int, visited: set):
        """递归打印树"""
        if node_id in visited or node_id not in self.nodes:
            return
        visited.add(node_id)
        node = self.nodes[node_id]

        prefix = '  ' * indent + ('├─ ' if indent > 0 else '')
        line = f"{prefix}[{node.id}] {node.operation}"
        if node.r2 > -1:
            line += f" (R²={node.r2:.4f})"
        if len(line) > 120:
            line = line[:117] + '...'
        lines.append(line)

        for pid in node.parent_ids:
            self._print_tree(pid, lines, indent + 1, visited)

    @staticmethod
    def _simplify_name(name: str) -> str:
        """标准化特征名称用于去重"""
        return name.replace(' ', '').replace('**', '^')

    def __len__(self):
        return len(self.nodes)