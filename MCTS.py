import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Optional
import random
import math
from dataclasses import dataclass

# =============================================================================
# 1. MONTE CARLO TREE SEARCH (MCTS) WITH KAN
# =============================================================================

@dataclass
class MCTSNode:
    """Node in the MCTS tree"""
    state: np.ndarray  # Board state
    parent: Optional['MCTSNode'] = None
    move: Optional[int] = None  # Move that led to this state
    children: Dict[int, 'MCTSNode'] = None
    
    # MCTS statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Prior probability from neural network
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}
    
    @property
    def value(self) -> float:
        """Average value from all visits"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def is_expanded(self) -> bool:
        """Check if node has been expanded"""
        return len(self.children) > 0


class MCTS:
    """
    Monte Carlo Tree Search with KAN evaluation.
    
    This is where KAN's superior evaluation is critical:
    - Better position evaluation → Better tree search
    - More accurate policy → Better move selection
    - Faster convergence → Less search needed
    """
    def __init__(
        self,
        model: nn.Module,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # Exploration constant
        self.temperature = temperature
        self.device = device
        self.model.eval()
    
    def search(self, board_state: np.ndarray, legal_moves: List[int]) -> np.ndarray:
        """
        Run MCTS from the given board state.
        
        Returns:
            move_probs: Improved policy after search
        """
        root = MCTSNode(state=board_state)
        
        # Expand root
        self._expand_node(root, legal_moves)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. SELECT: Traverse tree using UCB
            while node.is_expanded():
                move, node = self._select_child(node)
                search_path.append(node)
            
            # 2. EVALUATE: Use KAN to evaluate leaf
            value = self._evaluate_node(node)
            
            # 3. BACKUP: Propagate value up the tree
            self._backup(search_path, value)
        
        # Return improved policy based on visit counts
        return self._get_action_probs(root, legal_moves)
    
    def _expand_node(self, node: MCTSNode, legal_moves: List[int]):
        """
        Expand node using KAN's policy prediction.
        
        KAN advantage: More accurate prior probabilities
        → Better initial move selection
        → Faster convergence to strong moves
        """
        # Get KAN's policy prediction
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=-1)[0].cpu().numpy()
        
        # Mask illegal moves
        legal_probs = np.zeros_like(policy_probs)
        legal_probs[legal_moves] = policy_probs[legal_moves]
        
        # Normalize
        prob_sum = legal_probs.sum()
        if prob_sum > 0:
            legal_probs /= prob_sum
        else:
            # Uniform if all moves have zero probability
            legal_probs[legal_moves] = 1.0 / len(legal_moves)
        
        # Create child nodes
        for move in legal_moves:
            child_state = self._apply_move(node.state, move)
            child = MCTSNode(
                state=child_state,
                parent=node,
                move=move,
                prior=legal_probs[move]
            )
            node.children[move] = child
    
    def _select_child(self, node: MCTSNode) -> Tuple[int, MCTSNode]:
        """
        Select best child using UCB (Upper Confidence Bound).
        
        UCB = Q(s,a) + c_puct * P(s,a) * √(N(s)) / (1 + N(s,a))
        
        Where:
        - Q(s,a): Average value of taking action a
        - P(s,a): Prior probability (from KAN)
        - N(s): Parent visit count
        - N(s,a): Child visit count
        """
        best_score = -float('inf')
        best_move = None
        best_child = None
        
        for move, child in node.children.items():
            # UCB score
            q_value = child.value
            u_value = (
                self.c_puct * 
                child.prior * 
                math.sqrt(node.visit_count) / 
                (1 + child.visit_count)
            )
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        
        return best_move, best_child
    
    def _evaluate_node(self, node: MCTSNode) -> float:
        """
        Evaluate leaf node using KAN.
        
        KAN advantage: More accurate position evaluation
        → Better leaf evaluation
        → Better backpropagated values
        → Stronger overall search
        """
        state_tensor = torch.FloatTensor(node.state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value = self.model(state_tensor)
            return value.item()
    
    def _backup(self, search_path: List[MCTSNode], value: float):
        """Propagate value up the search path"""
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
    
    def _get_action_probs(self, root: MCTSNode, legal_moves: List[int]) -> np.ndarray:
        """
        Get improved policy after MCTS.
        
        Uses visit counts with temperature parameter.
        """
        visits = np.zeros(len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            if move in root.children:
                visits[i] = root.children[move].visit_count
        
        # Apply temperature
        if self.temperature == 0:
            # Deterministic: pick most visited
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            # Stochastic: proportional to visits^(1/T)
            visits_temp = visits ** (1 / self.temperature)
            probs = visits_temp / visits_temp.sum()
        
        return probs
    
    def _apply_move(self, state: np.ndarray, move: int) -> np.ndarray:
        """
        Apply move to board state.
        
        NOTE: This is a placeholder - you need to integrate
        your actual chess board update logic here.
        """
        # TODO: Use your existing board state update code
        new_state = state.copy()
        # ... apply move ...
        return new_state