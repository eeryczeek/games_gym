import random
import numpy as np
import torch
import players.EvaluateAI.SPG as SPG
from tqdm import tqdm, trange
import torch.nn.functional as F


class evaluateAIParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game)
                   for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = dict()
                for action in self.game.get_valid_moves(states[i]):
                    state_after_action = self.game.get_next_state(
                        np.copy(states[i]), action, player)
                    new_neutral_state = self.game.change_perspective(
                        state_after_action, self.game.get_opponent(player))

                    value = self.model(
                        torch.tensor(self.game.get_encoded_state(
                            new_neutral_state), device=self.model.device).unsqueeze(0)
                    )
                    action_probs[action] = self.game.get_opponent_value(
                        value.item()) / 2 + 0.5

                action_probs.update(
                    (key, value**self.args['temperature']) for key, value in action_probs.items())
                spg.memory.append((neutral_states[i], player))
                action = random.choices(
                    list(action_probs.keys()), weights=action_probs.values(), k=1)[0]
                spg.state = self.game.get_next_state(spg.state, action, player)
                value, is_terminal = self.game.get_value_and_terminated(
                    spg.state, action)
                if is_terminal:
                    for hist_neutral_state, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(
                            value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
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
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                new_memories = self.selfPlay()
                memory += new_memories

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(),
                       f"model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(),
                       f"optimizer_{iteration}_{self.game}.pt")
