import numpy as np
import torch
from games.tic_tac_toe import TicTacToe
from agents.evaluate_ai import evaluateAI


game = TicTacToe()
win1 = np.array([[0, 0, 1], [-1, 1, 0], [1, -1, 0]])
win2 = np.array([[1, 0, -1], [1, -1, 0], [1, 0, 0]])
win3 = np.array([[0, 0, 0], [0, -1, -1], [1, 1, 1]])
win4 = np.array([[1, 0, -1], [1, -1, 0], [0, 0, 0]])
win5 = np.array([[0, 0, 1], [-1, 0, 0], [1, -1, 0]])
wins = [win1, win2, win3, win4, win5]

draw1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
draw2 = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
draw3 = np.array([[-1, 1, -1], [-1, 1, -1], [1, -1, 1]])
draws = [draw1, draw2, draw3]

lose1 = np.array([[0, 0, -1], [1, -1, 0], [-1, 1, 0]])
lose2 = np.array([[-1, 0, -1], [1, 0, 0], [-1, 1, 0]])
lose3 = np.array([[-1, 0, 1], [-1, 0, 0], [-1, 1, 0]])
loses = [lose1, lose2, lose3]

for i, player in enumerate(players):
    print(f'model{i}')
    model = players[i].model
    model.eval()
    print('winning_possitions:')
    for winning_position in wins:
        encoded_state = game.get_encoded_state(winning_position)
        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        value = model(tensor_state)
        value = value.item()
        print(value)

    print('\ndrawing_possitions:')
    for drawing_position in draws:
        encoded_state = game.get_encoded_state(drawing_position)
        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        value = model(tensor_state)
        value = value.item()
        print(value)

    print('\nlosing_possitions:')
    for losing_position in loses:
        encoded_state = game.get_encoded_state(losing_position)
        tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

        value = model(tensor_state)
        value = value.item()
        print(value)
    print()
