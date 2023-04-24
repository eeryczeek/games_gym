import copy
import random
import torch
import numpy as np
from agents.evaluate_ai import evaluateAI


class ELO_Tournament_Trainer:
    '''Trains a model using ELO tournament method'''

    def __init__(self, game, args, model, optimizer, number_of_players=4, players=[]):
        self.game = game
        self.number_of_players = number_of_players
        self.players = players
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.model = model
        self.optimizer = optimizer

    def train_models(self, no_improvement=10):
        '''Trains the models using ELO tournament method'''
        while len(self.players) < self.number_of_players:
            player = evaluateAI(copy.deepcopy(
                self.model), copy.deepcopy(self.optimizer), self.game, self.args)
            player.learn()
            self.players.append(copy.deepcopy(player))

        return self.players

    def tournament(self):
        '''Plays a tournament between the models'''
        players_performances = {player: 0 for player in self.players}
        for player1 in self.players:
            for player2 in self.players:
                for _ in range(10):
                    result = self.play_game(player1, player2)
                    if result == 1:
                        players_performances[player2] -= 1
                    if result == -1:
                        players_performances[player1] -= 1
        return players_performances

    def play_game(self, player1, player2, debug=False):
        '''Plays a game between two models'''
        state = self.game.get_initial_state()
        player = 1
        while True:
            if debug:
                print(state)
            action = self.choose_action(state, player, player1.model) if player == 1 else self.choose_action(
                state, player, player2.model)
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(
                state, action)

            if is_terminal:
                if value == 1:
                    return player
                else:
                    return 0
            player = self.game.get_opponent(player)

    def choose_action(self, state, player, model):
        '''Chooses an action for a model
        Args:
            state: the current state of the game
            player: the player who is choosing the action
            model: the model that is choosing the action
        Returns:
            the action that the model chooses'''
        neutral_state = self.game.change_perspective(state, player)

        action_probs = dict()
        for action in self.game.get_valid_moves(state):
            state_after_action = self.game.get_next_state(
                np.copy(state), action, player)
            new_neutral_state = self.game.change_perspective(
                state_after_action, self.game.get_opponent(player))

            value = model(
                torch.tensor(self.game.get_encoded_state(
                    new_neutral_state), device=model.device).unsqueeze(0)
            )
            value = value.item()
            action_probs[action] = self.game.get_opponent_value(
                value) / 2 + 0.5

        if self.args['temperature'] == 'inf':
            return max(action_probs, key=action_probs.get)

        action_probs.update(
            (key, value**self.args['temperature']) for key, value in action_probs.items())
        return random.choices(list(action_probs.keys()), weights=action_probs.values(), k=1)[0]
