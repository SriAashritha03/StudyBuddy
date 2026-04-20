import numpy as np
import json
import os
import random

class StudyAssistantRL:
    def __init__(self, user_id='default_user', model_dir='backend/user_models'):
        self.user_id = user_id
        self.model_dir = model_dir
        self.states = ['Normal', 'Not in Good Mood', 'Yawning']
        self.actions = ['Continue studying', 'Take a break', 'Suggest relaxation activities']
        
        # RL Hyperparameters
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.8 # Discount factor
        self.epsilon = 0.3 # Exploration rate (higher initially)
        
        # Initialize Q-table: mapping states to dictionary of action-values
        self.q_table = {}
        for state in self.states:
            self.q_table[state] = {action: 0.0 for action in self.actions}
            
        self.load_q_table()
        
    def get_q_file_path(self):
        os.makedirs(self.model_dir, exist_ok=True)
        return os.path.join(self.model_dir, f"{self.user_id}_qtable.json")
        
    def load_q_table(self):
        filepath = self.get_q_file_path()
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    # Verify constraints
                    for state in self.states:
                        if state in data:
                            for action in self.actions:
                                if action in data[state]:
                                    self.q_table[state][action] = data[state][action]
                print(f"Loaded existing Q-Table for {self.user_id}")
            except Exception as e:
                print(f"Error loading Q-table: {e}")
                
    def save_q_table(self):
        filepath = self.get_q_file_path()
        try:
            with open(filepath, 'w') as f:
                json.dump(self.q_table, f, indent=4)
        except Exception as e:
            print(f"Error saving Q-table: {e}")
            
    def get_action(self, state):
        if state not in self.states:
            state = 'Normal'
            
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
            
        # Evaluate exploitation
        state_actions = self.q_table[state]
        best_action = max(state_actions, key=state_actions.get)
        
        # If all values are 0 (never explored here), pick uniformly rather than biasing to the first key
        if all(val == 0.0 for val in state_actions.values()):
            return random.choice(self.actions)
            
        return best_action
        
    def update(self, state, action, reward, next_state=None):
        if state not in self.states or action not in self.actions:
            return
            
        # Simplified Q-learning (if we don't have a specific `next_state` for immediate rewards, default it to current state)
        if next_state not in self.states:
            next_state = state
            
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        # Q-Learning update rule
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self.save_q_table()
        
        # Decay epsilon for more exploitation over time
        self.epsilon = max(0.05, self.epsilon * 0.98)
        
        print(f"RL Updated: State='{state}', Action='{action}', Reward={reward}")

if __name__ == "__main__":
    # Test script behavior
    agent = StudyAssistantRL('test_student')
    state = 'Yawning'
    action = agent.get_action(state)
    print(f"State: {state} => Recommended Action: {action}")
    # Simulating positive reward
    agent.update(state, action, reward=1)
