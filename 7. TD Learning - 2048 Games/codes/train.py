from board import board
from action import action
from episode import episode
from statistic import statistic
# from agent import player
from agent import rndenv
import sys

# History
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
HISTORY_DIR = 'history'
def save_history(data, file_name='history'):
    if '.pickle' not in file_name:
        file_name += '.pickle'
    # Store data (serialize)
    with open(os.path.join(HISTORY_DIR, file_name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_history(file_name='history'):
    if '.pickle' not in file_name:
        file_name += '.pickle'
    # Load data (deserialize)
    with open(os.path.join(HISTORY_DIR, file_name), 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data

def plot_hisotry(history, block=10, save=False):
    scores = history['score']
    if block > 0:
        scores = [np.mean(scores[low:low+block]) for low in range(0, len(scores), block)]
    plt.plot(scores)
    plt.xlabel("# %d game" %(block))
    if save:
        file_name = "%dk.png"%( len(scores) * block / 1000) 
        plt.savefig(os.path.join(HISTORY_DIR, file_name))
        plt.clf() 
        plt.cla()

from board import board
from action import action
from weight import weight
from array import array
from episode import episode
import random
import sys
import copy

class agent:
    """ base agent """
    
    def __init__(self, options = ""):
        self.info = {}
        options = "name=unknown role=unknown " + options
        for option in options.split():
            data = option.split("=", 1) + [True]
            self.info[data[0]] = data[1]
        return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return
    
    def open_episode(self, flag = ""):
        return
    
    def close_episode(self, flag = ""):
        return
    
    def take_action(self, state):
        return action()
    
    def check_for_win(self, state):
        return False
    
    def property(self, key):
        return self.info[key] if key in self.info else None
    
    def notify(self, message):
        data = message.split("=", 1) + [True]
        self.info[data[0]] = data[1]
        return
    
    def name(self):
        return self.property("name")
    
    def role(self):
        return self.property("role")
    
class random_agent(agent):
    """ base agent for agents with random behavior """
    
    def __init__(self, options = ""):
        super().__init__(options)
        seed = self.property("seed")
        if seed is not None:
            random.seed(int(seed))
        return
    
    def choice(self, seq):
        target = random.choice(seq)
        return target
    
    def shuffle(self, seq):
        random.shuffle(seq)
        return

    def close_episode(self, ep, flag = ""):
        return 

class learning_agent(agent):
    """ base agent for agents with a learning rate """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.alpha = 0.1
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = float(alpha)
        return

class rndenv(random_agent):
    """
    random environment
    add a new random tile to an empty cell
    2-tile: 90%
    4-tile: 10%
    """
    
    def __init__(self, options = ""):
        super().__init__("name=random role=environment " + options)
        return
    
    def take_action(self, state):
        empty = [pos for pos, tile in enumerate(state.state) if not tile]
        if empty:
            pos = self.choice(empty)
            tile = self.choice([1] * 9 + [2])
            return action.place(pos, tile)
        else:
            return action() 


class weight_agent(agent):
    """ base agent for agents with weight tables """
    
    def __init__(self, options = ""):
        super().__init__(options)
        self.episode = episode()
        self.net = []
        init = self.property("init")
        self.init_weights()
        if init is not None:
            self.init_weights(init)
        load = self.property("load")
        if load is not None:
            self.load_weights(load)
        self.alpha = 0.0025
        alpha = self.property("alpha")
        if alpha is not None:
            self.alpha = alpha
        return
    
    def __exit__(self, exc_type, exc_value, traceback):
        save = self.property("save")
        if save is not None:
            self.save_weights(save)
        return
    
    def init_weights(self):
        self.net += [weight(16**6)] # feature for line [0, 1, 2, 3, 4, 5] includes 16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [4, 5, 6, 7, 8, 9] includes 16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [0, 1, 2, 4, 5, 6] includes 16*16*16*16 possible
        self.net += [weight(16**6)] # feature for line [4, 5, 6, 8, 9, 10] includes 16*16*16*16 possible
        self.feature_idx = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6],[4, 5, 6, 8, 9, 10]]
        return

    def load_weights(self, path):
        input = open(path, 'rb')
        size = array('L')
        size.fromfile(input, 1)
        size = size[0]
        self.net = []
        for i in range(size):
            self.net += [weight()]
            self.net[-1].load(input)
            #print(self.net[-1][:3])
        return
    
    def save_weights(self, path):
        output = open(path, 'wb')
        array('L', [len(self.net)]).tofile(output)
        for w in self.net:
            w.save(output)
        return

    def open_episode(self, flag = ""):
        self.episode.clear()
        return

    def close_episode(self, ep, flag = ""):
        episode = ep[2:].copy()
        # backward
        episode.reverse()
        for i in range(1, len(episode)-1, 2):
            before_state_next, _, _, _ = episode[i-1]
            after_state, move, reward, _ = episode[i]
            before_state, _, _, _ = episode[i+1]
            #self.test_learn(before_state, move, reward, after_state, before_state_next)
            self.learn(before_state, move, reward, after_state, before_state_next)
        return

    def lineIndex(self, board_state):
        idxs = [0, 0, 0, 0]
        for f in range(4):
            for i in range(6):
                idxs[f] = idxs[f] * 16 + board_state[self.feature_idx[f][i]]
        return idxs

    def lineValue(self, board_state):
        value = 0.0
        for i in range(8):
            board = copy.copy(board_state)
            if (i >= 4):
                board.transpose()
            board.rotate(i)
            idxs = self.lineIndex(board)
            for f in range(4):
                value += self.net[f][idxs[f]]
        return value

    def updateLineValue(self, board_state, value):
        for i in range(8):
            board = copy.copy(board_state)
            if (i >= 4):
                board.transpose()
            board.rotate(i)
            idxs = self.lineIndex(board)
            for f in range(4):
                self.net[f][idxs[f]] += value
        return
    
    # CKW
    def compute_after_state(self, board_state, move):
        board_state = board(board_state)
        reward = move.apply(board_state)
        board_after_state = board(board_state)
        return board_after_state, reward
    
    def evaluate_state_action(self, board_state, op):
        move = action.slide(op)
        board_after_state, reward = self.compute_after_state(board_state, move)
        return reward + self.lineValue(board_after_state)
    
    def select_best_action(self, board_state):
        legal_ops = [op for op in range(4) if board(board_state).slide(op) != -1]
        if legal_ops:
            best_op = 0
            best_value = -1
            for op in legal_ops:
                value = self.evaluate_state_action(board_state, op)
                if value > best_value:
                    best_value = value
                    best_op = op
            return action.slide(best_op)
        else:
            return action()
        
    def learn(self, before_state, move, reward, after_state, before_state_next):
        move_next = self.select_best_action(before_state_next)
        after_state_next, reward_next = self.compute_after_state(before_state_next, move_next)
        reward_next = max(0, reward_next)
        TD_diff = reward_next + self.lineValue(after_state_next) - self.lineValue(after_state)
        self.updateLineValue(after_state, self.alpha*TD_diff/(4*8))
        
    def test_learn(self, before_state, move, reward, after_state, before_state_next):
        move_next = self.select_best_action(before_state_next)
        after_state_next, reward_next = self.compute_after_state(before_state_next, move_next)
        TD_diff = reward_next + self.lineValue(after_state_next) - self.lineValue(after_state)
        print("After State, V(s) = ", self.lineValue(after_state))
        print(after_state)
        print("Before State Next")
        print(before_state_next)
        print("After State Next, V(s) = ", self.lineValue(after_state_next), "Reward_next =", reward_next)
        print(after_state_next)
        print("Before Learn TD Difference = %.4f"%TD_diff)
        
        self.updateLineValue(after_state, self.alpha*TD_diff)
        
        TD_diff = reward_next + self.lineValue(after_state_next) - self.lineValue(after_state)
        
        print("After  Learn TD Difference = %.4f"%TD_diff)
        print("\n\n")





from tqdm import tqdm as tqdm
def train(play, evil, start=0, total=1000, block=0, limit=2000):
    play.train()
    n_games = total-start
    history = {'score':[]}
    pbar = tqdm(initial=start, total=total,  ascii=True)
    stat = statistic(n_games, block, limit)
    while not stat.is_finished():
        play.open_episode("~:" + evil.name())
        evil.open_episode(play.name() + ":~")
        stat.open_episode(play.name() + ":" + evil.name())
        
        game = stat.back()
        # Environment random pop up two tiles
        who = game.take_turns(evil, evil)
        move = who.take_action(game.state())
        game.apply_action(move)
        move = who.take_action(game.state())
        game.apply_action(move)
        while True:
            # Play and environment plays in turns
            who = game.take_turns(play, evil)
            move = who.take_action(game.state())
            if not game.apply_action(move) or who.check_for_win(game.state()):
                break
        win = game.last_turns(play, evil)
        stat.close_episode(win.name())
        play.close_episode(stat.back().ep_moves, win.name())
        evil.close_episode(win.name())
        
        history['score'] += [stat.back().ep_score]
        pbar.update()
    pbar.close()
    return history

def evaluate(play, evil, total=1000, block=0, limit=0):
    play.eval()
    pbar = tqdm(total=total)
    stat = statistic(total, block, limit)
    while not stat.is_finished():
        play.open_episode("~:" + evil.name())
        evil.open_episode(play.name() + ":~")
        stat.open_episode(play.name() + ":" + evil.name())
        
        game = stat.back()
        # Environment random pop up two tiles
        who = game.take_turns(evil, evil)
        move = who.take_action(game.state())
        game.apply_action(move)
        move = who.take_action(game.state())
        game.apply_action(move)
        while True:
            # Play and environment plays in turns
            who = game.take_turns(play, evil)
            move = who.take_action(game.state())
            if not game.apply_action(move) or who.check_for_win(game.state()):
                break
        win = game.last_turns(play, evil)
        stat.close_episode(win.name())
        play.close_episode(stat.back().ep_moves, win.name())
        evil.close_episode(win.name())
        pbar.update()
    return


class player(weight_agent): # should switch to weight_agent
    def __init__(self, options = ""):
        super().__init__("name=TD role=player " + options)
        self.alpha_train = 0.025
        return
    
    def take_action(self, state):
        return self.select_best_action(state)
    
    def train(self):
        self.alpha = self.alpha_train
    
    def eval(self):
        self.alpha = 0.0

start = 115 * 1000
total = 200 * 1000
interval = 1 * 1000

rand_environment = rndenv()
for begin in range(start, total, interval):
    end = begin + interval
    TD_player = player()
    if begin > 0:
        TD_player.load_weights("weights/%dk.pth"%(begin // 1000))
    history = train(TD_player, rand_environment, start=begin, total=end)
    if begin > 0:
        history_before = load_history("%dk"%(begin // 1000)) 
        history['score'] = history_before['score'] + history['score']
    TD_player.save_weights("weights/%dk.pth"%(end // 1000))
    save_history(history , "%dk"%(end // 1000))
    plot_hisotry(history, block=end//100, save=True)
