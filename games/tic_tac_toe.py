import numpy as np


class TicTacToe:
    '''TicTacToe game class'''

    def __init__(self):
        '''initializes the game

        Args
            row_count: number of rows
            column_count: number of columns
            action_size: number of actions
        '''
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        '''returns the name of the game'''
        return "TicTacToe"

    def get_initial_state(self):
        '''returns the initial state of the game'''
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        '''returns the next state of the game

        Args
            state: current state of the game
            action: action to be taken
            player: player to take the action

        Returns
            next state of the game
        '''
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state

    def get_valid_moves(self, state):
        '''returns a list of valid moves

        Args
            state: current state of the game

        Returns
            list of valid moves'''
        return (np.where(state.reshape(-1) == 0)[0]).astype(np.uint8)

    def check_win(self, state, action):
        '''checks if the last action resulted in a win

        Args
            state: current state of the game
            action: action to be taken

        Returns
            if the last action resulted in a win
        '''
        if action == None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
            np.sum(state[row, :]) == player * self.column_count
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        '''returns the value of the state and if the game is terminated

        Args
            state: current state of the game
            action: action to be taken

        Returns
            value of the state and if the game is terminated
        '''
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        '''returns the opponent of the player'''
        return -player

    def get_opponent_value(self, value):
        '''returns the value of the opponent'''
        return -value

    def change_perspective(self, state, player):
        '''changes the perspective of the state from the perspective of the player

        Args
            state: current state of the game
            player: player to change the perspective to

        Returns
            state from the perspective of the player
        '''
        return state * player

    def get_encoded_state(self, state):
        '''encodes the state into a form that can be used by the neural network

        Args
            state: current state of the game

        Returns
            encoded state
        '''
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
