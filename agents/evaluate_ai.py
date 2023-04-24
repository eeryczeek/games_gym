import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm


class evaluateAI:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

    def selfPlay(self):
        '''Plays a game against itself and returns the memory of the game

        Returns
            memory of the game
        '''
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)

            action_probs = dict()
            for action in self.game.get_valid_moves(state):
                state_after_action = self.game.get_next_state(
                    np.copy(state), action, player)
                new_neutral_state = self.game.change_perspective(
                    state_after_action, self.game.get_opponent(player))

                value = self.model(
                    torch.tensor(self.game.get_encoded_state(
                        new_neutral_state), device=self.model.device).unsqueeze(0)
                )
                action_probs[action] = self.game.get_opponent_value(value) + 1

            memory.append((neutral_state, player))
            action = random.choices(
                list(action_probs.keys()), weights=action_probs.values(), k=1)[0]
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(
                state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(
                        value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_outcome
                    ))
                return returnMemory
            player = self.game.get_opponent(player)

    def train(self, memory):
        '''Trains the model on the given memory

        Args
            memory: memory of the self play
        '''
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(
                len(memory) - 1, batchIdx + self.args['batch_size'])]
            state, value_targets = zip(*sample)

            state, value_targets = np.array(state), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32,
                                 device=self.model.device)
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device)

            out_value = self.model(state)

            value_loss = F.mse_loss(out_value, value_targets)
            loss = value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        '''Trains the model for the given number of iterations'''
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(),
                       f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(),
                       f"optimizer_{iteration}_{self.game}.pt")
