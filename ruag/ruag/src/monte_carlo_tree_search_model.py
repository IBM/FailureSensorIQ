from abc import ABC, abstractmethod
from collections import defaultdict
import math
        
class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=0.7):
        self.Q = defaultdict(int)  # total reward of each node
        self.N = defaultdict(int)  # total visit count for each node
        self.children = {}  # children of each node
        self.exploration_weight = exploration_weight

        self.path_rewards = [] 
        self.rules_rewards = defaultdict(list)  
        self.rules_last_reward = {}  
        self.reward_dict = {} 



    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"


        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children :
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.Q[n] / self.N[n]  # average reward

        return max(self.children[node], key=score)

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"

        path = self._select(node)
        leaf = path[-1]
        self._expand(leaf)
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            if node not in self.children or not self.children[node]:
                return path
            
            unexplored = set(self.children[node]) - set(list(self.children.keys()))
           
            if unexplored:
                n = unexplored.pop()
                path.append(n)
                return path

        
            node = self._uct_select(node)  # descend a layer deeper

    def _expand(self, node):
        "Update the `children` dict with the children of `node`"
        if node in self.children:
            return  # already expanded
        self.children[node] = node.find_children()
        return self.children[node]


    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = None
        while True:
            if node.is_terminal():
                reward = node.reward()
                self.reward_dict[frozenset(node.rule)] = reward
                return reward
            node = node.find_random_child()

    def _backpropagate(self, path, reward):
        "Send the reward back up to the ancestors of the leaf"
        for node in reversed(path):
            self.N[node] += 1
            self.Q[node] += reward
        leaf_node = path[-1]
        
        if leaf_node.is_terminal():  # 只有当路径到达叶节点时才记录
            rule_key = tuple(leaf_node.rule)
            self.rules_last_reward[rule_key] = reward

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(self.N[node])

        def uct(n): # n是节点
            "Upper confidence bound for trees"

            return self.Q[n] / self.N[n] + self.exploration_weight * math.sqrt(
                log_N_vertex / self.N[n]
            )
        
        return max(self.children[node], key=uct)

    def get_top_k_paths(self, k):
        sorted_paths = sorted(self.path_rewards, key=lambda x: x[1], reverse=True)
        return sorted_paths[:]

    def get_top_k_rules(self, k):
        avg_rewards = {rule: sum(rewards) / len(rewards) for rule, rewards in self.rules_rewards.items()}
        sorted_rules = sorted(avg_rewards.items(), key=lambda x: x[1], reverse=True)
        return sorted_rules[:k]

    def get_all_rules_rewards(self):
        # Calculate the average reward for each rule and return them
        rules_rewards = {}
        for rule, data in self.all_rules.items():
            if data["count"] > 0:  # Avoid division by zero
                avg_reward = data["reward"] / data["count"]
                rules_rewards[rule] = avg_reward
        return rules_rewards


    def get_all_rules_last_reward(self):
        sorted_rules = sorted(self.rules_last_reward.items(), key=lambda x: x[1], reverse=True)

        return sorted_rules
    
    def get_all_rules_all_simulated_path(self):
        # remove zero ones
        sorted_rules = sorted(self.reward_dict.items(), key=lambda x: x[1], reverse=True)
        # sorted_rules = [x for x in sorted_rules if x[1] > 0]
        return sorted_rules

    def log(self, ):
        node2reward = {tuple(k.rule): v for k, v in self.Q.items()}
        node2visit = {tuple(k.rule): v for k, v in self.N.items()}
        node_info = {k: (node2reward.get(k, 0), node2visit.get(k, 0)) for k in node2reward.keys()}
        return node_info
    
class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"

        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True