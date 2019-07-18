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
from episode import episode


class statistic:
    """ container & statistic of episodes """
    
    def __init__(self, total, block = 0, limit = 0):
        """
        the total episodes to run
        the block size of statistic
        the limit of saving records
        
        note that total >= limit >= block
        """
        self.total = total
        self.block = block if block else total
        self.limit = limit if limit else total
        self.data = []
        self.count = 0
        return
    
    def show(self, tstat = True):
        """
        show the statistic of last 'block' games
        
        the format would be
        1000   avg = 273901, max = 382324, ops = 241563 (170543|896715)
               512     100%   (0.3%)
               1024    99.7%  (0.2%)
               2048    99.5%  (1.1%)
               4096    98.4%  (4.7%)
               8192    93.7%  (22.4%)
               16384   71.3%  (71.3%)
        
        where (block = 1000 by default)
         '1000': current index (n)
         'avg = 273901': the average score is 273901
         'max = 382324': the maximum score is 382324
         'ops = 241563 (170543|896715)': the average speed is 241563
                                         the average speed of player is 170543
                                         the average speed of environment is 896715
         '93.7%': 93.7% (937 games) reached 8192-tiles (a.k.a. win rate of 8192-tile)
         '22.4%': 22.4% (224 games) terminated with 8192-tiles (the largest)
        """
        blk = min(len(self.data), self.block)
        stat = [0] * 64
        sop, pop, eop = 0, 0, 0
        sdu, pdu, edu = 0, 0, 0
        ssc, msc = 0, 0
        for i in range(1, blk + 1):
            ep = self.data[-i]
            ssc += ep.score()
            msc = max(ep.score(), msc)
            stat[max(ep.state().state)] += 1
            sop += ep.step()
            pop += ep.step(action.slide.type)
            eop += ep.step(action.place.type)
            sdu += ep.time()
            pdu += ep.time(action.slide.type)
            edu += ep.time(action.place.type)
        
        print("%d\t" "avg = %d, max = %d, ops = %d (%d|%d)" % (self.count, ssc / blk, msc, sop * 1000 / sdu, pop * 1000 / pdu, eop * 1000 / edu))
        
        if not tstat:
            return
        c = 0
        for t in range(0, len(stat)):
            if c >= blk:
                break
            if not stat[t]:
                continue
            accu = sum(stat[t:])
            print("\t" "%d" "\t" "%s%%" "\t" "(%s%%)" % ((1 << t) & -2, accu * 100 / blk, stat[t] * 100 / blk)) # type, win rate, % of ending
            c += stat[t]
        
        print()
        return
    
    def summary(self):
        block = self.block
        self.block = len(self.data)
        self.show()
        self.block = block
        return
    
    def is_finished(self):
        return self.count >= self.total
    
    def open_episode(self, flag = ""):
        if self.count >= self.limit:
            self.data = self.data[1:]
        self.count += 1
        self.data += [episode()]
        self.data[-1].open_episode(flag)
        return
    
    def close_episode(self, flag = ""):
        self.data[-1].close_episode(flag)
        if self.count % self.block == 0:
            self.show()
        return
    
    def at(self, i):
        return self.data[i]
    
    def front(self):
        return self.data[0]
    
    def back(self):
        return self.data[-1]
    
    def save(self, output):
        """ serialize this action to a file object """
        output.write(self.__str__())
        return True
    
    def load(self, input):
        """ deserialize from a file object """
        self.data = []
        while True:
            # load an episode
            ep = episode()
            if ep.load(input):
                self.data += [ep]
            else:
                break
        self.total = max(self.total, len(self.data))
        self.count = len(self.data)
        return True
    
    def __str__(self):
        return "\n".join([str(ep) for ep in self.data]) + "\n"
    
    
if __name__ == '__main__':
    print('2048 Demo: statistic.py\n')
    pass
