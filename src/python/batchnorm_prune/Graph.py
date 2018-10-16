import numpy as np
import logging, sys


class Node(object):
    def __init__(self, i=None, o=None):
        if i is None:
            self.i = set()
        else:
            self.i = i
        if o is None:
            self.o = set()
        else:
            self.o = o

class Flow(object):
    """ basic direct graph structures
    """            
    def __init__(self, nn=None):
        """ Return a Flow object"""
        self.nodes = dict()
        self.terminals = set()
        self.edgename_map = dict()
        self.flow = dict()
        self.cost = dict()
        self.input_nodes = set()
        self.zero_nodes = set()
        
    def add_node(self, id):
        """ Add a node to flow """
        if id in self.nodes:
            pass
        else:
            self.nodes[id] = Node()

    def add_input_node(self, id):
        """ Add a input node (no inlink) """
        self.add_node(id)
        if id not in self.input_nodes:
            self.input_nodes.add(id)

    def add_terminal(self, id):
        """ Add a terminal node (no outlink) """
        self.add_node(id)
        if id not in self.terminals:
            self.terminals.add(id)
        
    def add_edge(self, left, right):
        """ Add an edge between two nodes """
        try:
            assert left != right
            self.nodes[left].o.add(right)
            self.nodes[right].i.add(left)
        except KeyError:
            print "Key Not Found:", left, "or", right

    def export_dot(self, filename):
        """ Export dot script for visualization """
        print "strict digraph {"
        print "rankdir=\"LR\""
        for s in self.nodes.keys():
            for t in self.nodes[s].o:
                print s.replace('/', '__'), " -> ", t.replace('/', '__')
        print "}"

    def print_summary(self):
        print "number of nodes: ", len(self.nodes)
        print "number of zero nodes: ", len(self.zero_nodes)
        print "number of edges: ", sum([len(n.o) for n in self.nodes.values()])

    def shadow_snapshot():
        new = Flow()
        new.nodes = self.nodes.copy()
        new.terminals = self.terminals.copy()
        return new


class NeuralFlow(Flow):
    def __init__(self, nn=None):
        Flow.__init__(self, nn)
        self.variable_map=dict()
        self.constraint_map=dict()
    def prune(self):
        has_node_pruned = False
        to_del = []
        for id in self.nodes.keys():            
            if id not in self.terminals:
                if not self.nodes[id].o:
                    for inode in self.nodes[id].i:
                        self.nodes[inode].o.remove(id)
                    has_node_pruned = True
                    to_del.append(id)
        if has_node_pruned:
            for id in to_del:
                del self.nodes[id]
            self.prune() # recursively prune other nodes
        else:
            self.zero_nodes = set()
            for id in self.nodes.keys():
                if not self.nodes[id].i and id not in self.input_nodes:
                    stack = [id]
                    while stack:
                        next_id = stack.pop()
                        self.zero_nodes.add(next_id)
                        for v in self.nodes[next_id].o:
                            if all([w in self.zero_nodes for w in self.nodes[v].i]):
                                stack.append(v)    
    def add_suffix(self, suffix):
        """
        add suffix to all keys
        """
        nodes = dict()
        for key, v in self.nodes.items():
            i = set()
            for s in v.i:
                i.add(suffix + s)
            v.i = i
            o = set()
            for t in v.o:
                o.add(suffix + t)
            v.o = o
            nodes[suffix + key] = v
        self.nodes = nodes

        terminals = set()
        for key in self.terminals:
            terminals.add(suffix + key)
        self.terminals = terminals

        zero_nodes = set()
        for key in self.zero_nodes:
            zero_nodes.add(suffix + key)
        self.zero_nodes = zero_nodes

        self.edgename_map = dict()
        self.flow = dict()
    def compute_node_visits(self):
        import copy
        visits = 0
        for t in self.terminals:
            a = set([t])
            b = set()
            while a:
                c = set()
                d = copy.copy(b)
                for s in a:                    
                    for t in self.nodes[s].i:
                        b.add(t)
                        c.add(t)
                a = c - d
            visits = visits + len(b)
        return visits
    def get_channel_pruned(self, varname, v):
        if len(self.edgename_map[varname]) == 2:
            v_in, v_out = self.edgename_map[varname]
            for i in range(v.shape[0]):
                if v[i]:
                    v1 = v_in  + "_%d" % i
                    v2 = v_out + "_%d" % i
                    self.nodes[v1].o.remove(v2)
                    self.nodes[v2].i.remove(v1)        
            
        elif len(self.edgename_map[varname]) == 3:
            v_in1, v_in2, v_out = self.edgename_map[varname]
            for i in range(v.shape[0]):
                if v[i]:
                    v1 = v_in1  + "_%d" % i
                    v2 = v_out + "_%d" % i
                    self.nodes[v1].o.remove(v2)
                    self.nodes[v2].i.remove(v1)        
            
                    v1 = v_in2  + "_%d" % i
                    v2 = v_out + "_%d" % i
                    self.nodes[v1].o.remove(v2)
                    self.nodes[v2].i.remove(v1)        
            
        return True
