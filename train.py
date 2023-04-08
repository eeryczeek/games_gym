import torch
from agents.res_net import ResNet
from agents.trainer import ELO_Tournament_Trainer
from games.tic_tac_toe import TicTacToe


game = TicTacToe()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(game, 8, 8, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
args = {
    'num_iterations': 1,
    'num_selfPlay_iterations': 16,
    'num_parallel_games': 4,
    'num_epochs': 1,
    'batch_size': 128,
    'temperature': 1,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}
trainer = ELO_Tournament_Trainer(game, device, args, model, optimizer)
players = trainer.train_models()
