#!/usr/bin/env python3

"""
Basic framework for developing 2048 programs in Python

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
Modifier: Kuo-Hao Ho (lukewayne123)
"""

from board import board
from action import action
import time
import io


class episode:
    """ container of actions and time usages of an episode """
    
    def __init__(self):
        self.clear()
        return
    
    def state(self):
        return self.ep_state
    
    def score(self):
        return self.ep_score
    
    def open_episode(self, tag = ""):
        self.ep_open = tag, self.millisec()  # flag, time usage
        return
    
    def close_episode(self, tag = ""):
        self.ep_close = tag, self.millisec()  # flag, time usage
        return
    
    def apply_action(self, move):
        reward = move.apply(self.state())
        state = board(self.state())
        if reward == -1:
            return False
        usage = self.millisec() - self.ep_time
        self.ep_state = state
        record = board(self.state()), move, reward, usage # state, action, reward, time usage
        self.ep_moves += [record]
        #self.ep_records += [(board(state), move, reward, usage)]
        self.ep_score += reward
        return True
    
    def take_turns(self, play, evil):
        self.ep_time = self.millisec()
        if max(self.step() + 1, 2) % 2 != 0:
            return play
        else:
            return evil
    
    def last_turns(self, play, evil):
        return self.take_turns(evil, play)
    
    def step(self, who = -1):
        size = len(self.ep_moves)
        if who == action.slide.type:
            return int((size - 1) / 2)
        if who == action.place.type:
            return size - int((size - 1) / 2)
        return size

    def time(self, who=-1):
        if self.ep_moves:
            if who == action.slide.type:
                return sum(
                    [mv[3] for mv in self.ep_moves[slice(2, self.step(), 2)]])  # state, action, reward, time usage
            if who == action.place.type:
                return self.ep_moves[0][3] + sum(
                    [mv[3] for mv in self.ep_moves[slice(1, self.step(), 2)]])  # state, action, reward, time usage
        return self.ep_close[1] - self.ep_open[1]  # flag, time usage

    def actions(self, who=-1):
        if self.ep_moves:
            if who == action.slide.type:
                return [mv[1] for mv in self.ep_moves[slice(2, self.step(), 2)]]  # state, action, reward, time usage
            if who == action.place.type:
                return [self.ep_moves[0][1]] + [mv[1] for mv in self.ep_moves[
                    slice(1, self.step(), 2)]]  # state, action, reward, time usage
        return [mv[1] for mv in self.ep_moves]  # state, action, reward, time usage
        
    def save(self, output):
        """ serialize this episode to a file object """
        output.write(self.__str__())
        return True
    
    def load(self, input):
        """ deserialize from a file object """
        try:
            self.clear()
            line = input.readline()
            # line --> open|moves|close
            delim = line.index("|"), line.index("|", line.index("|") + 1)
            open = line[0:delim[0]]
            close = line[(delim[1] + 1):]
            moves = line[(delim[0] + 1):delim[1]]
            # open --> flag@time
            delim = open.index("@")
            self.ep_open = open[0:delim], int(open[(delim + 1):])
            # close --> flag@time
            delim = close.index("@")
            self.ep_close = close[0:delim], int(close[(delim + 1):])
            # moves --> action[reward](time)...
            minput = io.StringIO(moves)
            while True:
                # check if EOF
                ipt = minput.tell()
                if minput.read(1) == "":
                    break
                minput.seek(ipt)
                # ?? --> action
                a = action.parse(minput)
                self.ep_score += a.apply(self.ep_state)
                # [?] --> reward
                r = self.load_optional_value(minput, "[]")
                # (?) --> time
                t = self.load_optional_value(minput, "()")
                # (action, reward, time)
                self.ep_moves += [(a, r, t)]
            return True
        except (RuntimeError, ValueError, IndexError):
            pass
        return False
    
    def load_optional_value(self, minput, flag):
        t = 0
        ipt = minput.tell()
        if minput.read(1) == flag[0]:
            buf = minput.read()
            t = buf[0:buf.index(flag[-1])]
            minput.seek(ipt + len(t) + 2)
            t = int(t)
        else:
            minput.seek(ipt)
            t = 0
        return t
    
    def __str__(self):
        open = str(self.ep_open[0]) + "@" + str(self.ep_open[1])
        moves = "".join([str(m[0]) + ("[" + str(m[1]) + "]" if m[1] else "") + ("(" + str(m[2]) + ")" if m[2] else "") for m in self.ep_moves])
        close = str(self.ep_close[0]) + "@" + str(self.ep_close[1])
        return open + "|" + moves + "|" + close
    
    def clear(self):
        self.ep_state = self.initial_state()
        self.ep_score = 0
        self.ep_time = 0
        self.ep_moves = []
        self.ep_records = []
        self.ep_open = "N/A", 0 # flag, time usage
        self.ep_close = "N/A", 0 # flag, time usage
        return
    
    def initial_state(self):
        return board()
    
    def millisec(self):
        return int(round(time.time() * 1000))
        
    
if __name__ == '__main__':
    print('2048 Demo: episode.py\n')
    pass
