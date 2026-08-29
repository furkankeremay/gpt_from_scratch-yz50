
from graphviz import Digraph
import math

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
    def __repr__(self):
        return f"Value = ({self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
            out = Value(self.data * other.data, (self, other), '*')
            return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        return out

def lol():
    h = 0.001
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(5.0, label='c')
    d = a*b ;  d.label = 'd'
    e = d+c ; e.label = 'e'
    f = Value(-2.0, label='f')
    L2 = e*f ; L.label = 'L'

    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(5.0, label='c')
    d = a*b ;  d.label = 'd'
    e = d+c ; e.label = 'e'
    f = Value(-2.0, label='f')
    L1 = e*f ; L.label = 'L'

    print((L2.data-L1.data)/h)

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)
            
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
    return dot

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(5.0, label='c')
d = a*b ;  d.label = 'd'
e = d+c ; e.label = 'e'
f = Value(-2.0, label='f')
L = e*f ; L.label = 'L'

e.grad = -2.0
f.grad = -1.0
L.grad = 1.0
d.grad = -2.0
c.grad = -2.0
a.grad = 6.0
b.grad = -4.0

# draw_dot(L).render('islem_agaci', format='png', view=True)


x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')

w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')

b = Value(6.8813735870195432, label='b')

x1w1 = x1 * w1; x1w1.label = 'x1*w1'
x2w2 = x2 * w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'


o.grad = 1.0

n.grad = 1.0 - (o.data**2) 

x1w1x2w2.grad = n.grad
b.grad = n.grad

x1w1.grad = x1w1x2w2.grad
x2w2.grad = x1w1x2w2.grad

x1.grad = w1.data * x1w1.grad
w1.grad = x1.data * x1w1.grad

x2.grad = w2.data * x2w2.grad
w2.grad = x2.data * x2w2.grad

draw_dot(o).render('neuron_agaci', format='png', view=True)
