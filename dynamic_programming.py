#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

# Tom van Gelooven, Lisa Schouten
# s1853686, s3162915

import numpy as np
from world import World


class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self, env, gamma=1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)
    
        # IMPLEMENT YOUR VALUE ITERATION ALGORITHM HERE
        while(True): # loop
            delta = 0 # error variable
            copyV_s = V_s # copy the state value table
            for index, value in enumerate(copyV_s): # iterate over values of the copy
                action_eval = [] # empty list
                for action in env.actions:
                    s_prime, r = env.transition_function(index, action) # calculate new state and reward from transition function
                    action_eval.append(r + gamma * V_s[s_prime]) #append to list
                V_s[index] = max(action_eval) # set new state value to maximal value
                delta = max([delta, np.abs(value - V_s[index])]) # calculate error
            print(delta)
            if delta < theta: # if maximal error in all states is smaller than theta
                break # break loop
        self.V_s = V_s # update V_s
        # print(V_s)
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        # IMPLEMENT YOUR Q-VALUE ITERATION ALGORITHM HERE

        while(True):
            delta = 0 # error variable
            for state in env.states: # loop over possible states
                for action_no in range(env.n_actions): # loop over possible actions
                    x = Q_sa[state, action_no] # copy old Q-value
                    s_prime, r = env.transition_function(state, env.actions[action_no]) # caculate s' and r from T(s,a)
                    state_eval = [] # empty list
                    for action_prime in range(env.n_actions): # loop over actions
                        state_eval.append(Q_sa[s_prime, action_prime]) #append Q(s', a')
                    Q_sa[state, action_no] = r + gamma*max(state_eval) # set Q(s,a) to the one with maximum Q-value from action list
                    delta = max(delta, np.abs(x - Q_sa[state, action_no])) # calculate error
            if delta < theta: #if error is small enough
                break
        self.Q_sa = Q_sa # update Q-value table
        # print(Q_sa)
        return
                
    def execute_policy(self, env, table='V'):
        # Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state()  # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                # IMPLEMENT ACTION VALUE ESTIMATION FROM self.V_s HERE !!!
                action_eval = [] # empty list
                for action in available_actions: # loop over possible actions
                    s_prime, r = env.transition_function(current_state, action) # calculate new state and reward from transition function
                    action_eval.append([r + self.V_s[s_prime], action]) # append value from state value table and action string

                greedy_action = max(action_eval, key=lambda item: item[0])[1]  # select the action with maximal value

            elif table == 'Q' and self.Q_sa is not None:
                # IMPLEMENT ACTION VALUE ESTIMATION FROM self.Q_sa here !!!
                action_eval = [] # empty list
                for index, action in enumerate(available_actions): # loop over possible actions
                    action_eval.append([self.Q_sa[current_state, index], action]) # append Q-value for action
                greedy_action = max(action_eval, key=lambda item: item[0])[1] # select action with maximal Q-value
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None

            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))


if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy

    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

