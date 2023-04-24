import numpy as np


class ConnectFour:
    '''ConnectFour game class'''

    def __init__(self):
        '''Initializes the game

        Args
            row_count: number of rows
            column_count: number of columns
            action_size: number of actions
            in_a_row: number of pieces in a row to win
        '''
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self) -> str:
        '''Returns the name of the game'''
        return "ConnectFour"

    def get_initial_state(self) -> np.ndarray:
        '''Returns the initial state of the game'''
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state, action, player) -> np.ndarray:
        '''Returns the next state of the game

        Args
            state: current state of the game
            action: action to be taken
            player: player to take the action

        Returns
            next state of the game
        '''
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state

    def get_valid_moves(self, state) -> np.ndarray:
        '''Returns a list of valid moves

        Args
            state: current state of the game

        Returns
            list of valid moves
        '''
        return (state[0] == 0).astype(np.uint8)

    def check_win(self, state: np.ndarray, action: int) -> bool:
        '''Checks if the last action resulted in a win

        Args
            state (np.ndarray): current state of the game
            action (int): action to be taken

        Returns
            True if the last action resulted in a win, False otherwise
        '''
        if action == None:
            return False

        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row: int, offset_column: int) -> int:
            '''Counts the number of pieces in a row in a given direction

            Args
                offset_row (int): row offset
                offset_column (int): column offset

            Returns
                number of pieces in a row in a given direction
            '''
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

    def get_value_and_terminated(self, state, action) -> tuple[int, bool]:
        '''Returns the value of the state and whether the game is terminated

        Args
            state: current state of the game
            action: action to be taken

        Returns
            value of the state and whether the game is terminated
        '''
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player) -> int:
        '''Returns the opponent of the player

        Args
            player: current player

        Returns
            opponent of the player
        '''
        return -player

    def get_opponent_value(self, value) -> int:
        '''Returns the value of the opponent

        Args
            value: value of the player

        Returns
            value of the opponent
        '''
        return -value

    def change_perspective(self, state, player) -> np.ndarray:
        '''Changes the perspective of the state

        Args
            state (np.array): current state of the game
            player (int): current player

        Returns
            np.array: state with perspective of the player
        '''
        return state * player

    def get_encoded_state(self, state) -> np.array:
        '''Encodes the state into a form that can be used by the neural network

        Args
            state (np.array): current state of the game

        Returns
            np.array: encoded state
        '''
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)

        if len(state.shape) == 3:
            encoded_state = np.swapaxes(encoded_state, 0, 1)

        return encoded_state
