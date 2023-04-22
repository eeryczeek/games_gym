import numpy as np


class ConnectFour:
    def __init__(self):
        '''initializes the game
        :param row_count: number of rows
        :param column_count: number of columns
        :param action_size: number of actions
        :param in_a_row: number of pieces in a row to win'''
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        '''returns the name of the game'''
        return "ConnectFour"

    def get_initial_state(self):
        '''returns the initial state of the game'''
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player):
        '''returns the next state of the game
        :param state: current state of the game
        :param action: action to be taken
        :param player: player to take the action
        :return: next state of the game'''
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state):
        '''returns a list of valid moves
        :param state: current state of the game
        :return: list of valid moves'''
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state, action):
        '''checks if the last action resulted in a win
        :param state: current state of the game
        :param action: action to be taken
        :return: True if the last action resulted in a win, False otherwise'''
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            '''counts the number of pieces in a row in a given direction
            :param offset_row: row offset
            :param offset_column: column offset
            :return: number of pieces in a row in a given direction'''
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0
                    or r >= self.row_count
                    or c < 0
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
        )

    def get_value_and_terminated(self, state, action):
        '''returns the value of the state and whether the game is terminated
        :param state: current state of the game
        :param action: action to be taken
        :return: value of the state and whether the game is terminated'''
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        '''returns the opponent of the player
        :param player: current player
        :return: opponent of the player'''
        return -player

    def get_opponent_value(self, value):
        '''returns the value of the opponent
        :param value: value of the player
        :return: value of the opponent'''
        return -value

    def change_perspective(self, state, player):
        '''changes the perspective of the state
        :param state: current state of the game
        :param player: player to take the action
        :return: state from the perspective of the player'''
        return state * player

    def get_encoded_state(self, state):
        '''encodes the state into a form that can be used by the neural network
        :param state: current state of the game
        :return: encoded state'''
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
