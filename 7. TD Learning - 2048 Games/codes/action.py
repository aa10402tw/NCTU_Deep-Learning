#!/usr/bin/env python3

"""
Basic framework for developing 2048 programs in Python

Author: Hung Guei (moporgic)
        Computer Games and Intelligence (CGI) Lab, NCTU, Taiwan
        http://www.aigames.nctu.edu.tw
Modifier: Kuo-Hao Ho (lukewayne123)
"""

from board import board


class action:
    """ base action """
    
    def __init__(self, code = -1):
        self.code = code
        return
    
    def apply(self, state):
        """ apply this action to a specific board object """
        return -1
    
    def save(self, output):
        """ serialize this action to a file object """
        output.write(self.__str__())
        return True
    
    def load(self, input):
        """ deserialize from a file object """
        input.read(2)
        return False
    
    def __str__(self):
        return "??"
    
    def event(self):
        return self.code & 0x00ffffff
    
    def type(self):
        return self.code & 0xff000000
    
action.prototype = []
def parse(input):
    for proto in action.prototype:
        a = proto()
        if a.load(input):
            return a
    input.read(2)
    return action()
action.parse = parse
        
class slide(action):
    """ create a sliding action with board opcode """
    
    type = 0x73000000
    res = [ "#U", "#R", "#D", "#L", "#?" ]
    
    def __init__(self, code = -1):
        super().__init__(slide.type | code)
        return
    
    def apply(self, state):
        return state.slide(self.event())
    
    def __str__(self):
        return slide.res[max(min(self.event(), 4), 0)]
    
    def load(self, input):
        ipt = input.tell()
        val = input.read(2)
        code = slide.res.index(val) if val in slide.res else -1
        if code >= 0 and code < 4:
            self.code = slide(code).code
            return True
        input.seek(ipt)
        return False

action.slide = slide
action.prototype += [action.slide]

        
class place(action):
    """ create a placing action with position and tile """
    
    type = 0x70000000
    res = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ?")
    
    def __init__(self, pos = -1, tile = -1):
        super().__init__(place.type | (pos & 0x0f) | (tile << 4))
        return
    
    def position(self):
        return self.event() & 0x0f
    
    def tile(self):
        return self.event() >> 4
    
    def apply(self, state):
        return state.place(self.position(), self.tile())
    
    def __str__(self):
        return place.res[self.position()] + place.res[max(min(self.tile(), 36), 0)]
    
    def load(self, input):
        ipt = input.tell()
        val = input.read(2)
        pos = place.res.index(val[0]) if val[0] in place.res else -1
        tile = place.res.index(val[1]) if val[1] in place.res else -1
        if pos >= 0 and pos < 16 and tile > 0 and tile < 36:
            self.code = place(pos, tile).code
            return True
        input.seek(ipt)
        return False

action.place = place
action.prototype += [action.place]


if __name__ == '__main__':
    print('2048 Demo: action.py\n')
    pass
