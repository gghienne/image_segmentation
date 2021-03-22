import numpy as np

class KolmogorovSolver:

    def __init__(self, Graph, SOURCE, TAP):
        self.TR_S = 1
        self.TR_T = 2
        self.TR_NULL = 0
        self.SOURCE = SOURCE
        self.TAP = TAP
        self.Graph = np.array(Graph)
        self.Nvertex = self.Graph.shape[0]
        self.Nindex = np.arange(self.Nvertex)

        self.Flow = np.zeros(self.Graph.shape).astype(int)
        self.resFlow=self.Graph.copy()

        self.Tree = np.ones((self.Nvertex)).astype(int) * self.TR_NULL
        self.Tree[SOURCE] = self.TR_S
        self.Tree[TAP] = self.TR_T
        # print(self.Tree)

        self.Parent = np.zeros(self.Graph.shape[0]).astype(int) - 1

        self.Neighbours = [self.getNeighbours(i) for i in range(self.Nvertex)]

        self.Actives = [False for i in range(self.Nvertex)]
        self.Actives[self.SOURCE] = True
        self.Actives[self.TAP] = True
        self.Actq = [self.TAP, self.SOURCE]
        self.Orphans = set()

        self.source_group = []
        self.tap_group = []
        self.cuts = []
        self.maxflow = 0

    def trackOrigin(self, x):
        Visited = [False] * self.Nvertex
        cur = x
        while (self.Parent[cur] != -1):
            if Visited[self.Parent[cur]]:
                return -1
            Visited[cur] = True
            cur = self.Parent[cur]
        return cur

    def bckTrackOrig(self, x):
        path = [x]
        Visited = [False] * self.Nvertex
        cur = x
        while (self.Parent[cur] != -1):
            if Visited[self.Parent[cur]]:
                return -1
            Visited[cur] = True
            cur = self.Parent[cur]
            path.append(cur)
        return path

    def createPath(self, p, q):
        if self.Tree[p] == self.TR_S and self.Tree[q] == self.TR_T:
            # print("Creating path with ",self.bckTrackOrig(p), "and ",self.bckTrackOrig(q)," : ",list(reversed(self.bckTrackOrig(p)))+self.bckTrackOrig(q))
            return list(reversed(self.bckTrackOrig(p))) + self.bckTrackOrig(q)
        elif self.Tree[q] == self.TR_S and self.Tree[p] == self.TR_T:
            # print("Creating path with ",self.bckTrackOrig(p), "and ",self.bckTrackOrig(q)," : ",list(reversed(self.bckTrackOrig(q)))+self.bckTrackOrig(p))
            return list(reversed(self.bckTrackOrig(q))) + self.bckTrackOrig(p)
        else:
            return None

    def getNeighbours(self, Node):
        outwards = self.Graph[Node, :]
        inwards = self.Graph[:, Node]

        outwards = self.Nindex[outwards != 0]
        inwards = self.Nindex[inwards != 0]

        return list(set(outwards).union(set(inwards)))

    def PathAugmflow(self, path):
        sat = []
        dt = np.inf
        for i in range(len(path) - 1):
            f = self.Flow[path[i], path[i + 1]]
            c = self.Graph[path[i], path[i + 1]]
            # print(c)
            if c - f < dt:
                dt = c - f
        for i in range(len(path) - 1):
            self.Flow[path[i], path[i + 1]] += dt
            self.Flow[path[i+1], path[i ]] -= dt
            if self.Flow[path[i], path[i + 1]] == self.Graph[path[i], path[i + 1]]:
                sat.append([path[i], path[i + 1]])
        return dt, sat

    def tree_cap(self, p, q):
        if self.Tree[p] == self.TR_S:
            return self.Graph[p, q] - self.Flow[p, q]
        # elif self.Tree[p]==self.TR_T:
        else:
            return self.Graph[q, p] - self.Flow[q, p]


    def tree_cap_inv(self, q, p):
        if self.Tree[p] == self.TR_S:
            return self.Graph[q, p] - self.Flow[q, p]
        else:
            return self.Graph[p, q] - self.Flow[p, q]


    def printTree(self):
        asignations = {}
        for idd, v in enumerate(self.Tree):
            if v == self.TR_S:
                asignations[idd] = "S"
            elif v == self.TR_T:
                asignations[idd] = "T"
            else:
                asignations[idd] = "n"
        # print("Current tree:",asignations)

    def printParents(self):
        asignations = {}
        for idd, v in enumerate(self.Parent):
            asignations[idd] = v
        print("Current parents:",asignations)



    def grow_stage(self):
        #print("-------------Growth stage------------")
        while len(self.Actq) != 0:
            # print("Current actives0",self.Actives)
            p = self.Actq[0]
            #self.Actq.pop(0)
            if self.Actives[p]:

                #self.Actq.append(p)
                # print("Current p: ",p)
                neigh_p = np.array(self.Neighbours[p])
                # dist=np.array([self.tree_cap(p,q) for q in neigh_p])
                # neigh_p=neigh_p[np.argsort(dist)]
                # print("Neigbourhood: ",neigh_p)
                for q in neigh_p:
                    # print("Current neigbour: ",q)
                    if self.tree_cap(p, q) > 0:
                        if self.Tree[q] == self.TR_NULL:
                            self.Tree[q] = self.Tree[p]
                            self.Parent[q] = p
                            self.Actq.append(q)
                            self.Actives[q] = True
                            # print("Current actives1",self.Actives)
                        if self.Tree[q] != self.TR_NULL and self.Tree[q] != self.Tree[p]:
                            # print("Current actives 2",self.Actives)
                            return self.createPath(p, q)
            self.Actq.pop(0)
            self.Actives[p] = False

        return None

    def augment_stage(self, path):
        #print("-------------Augment stage------------")

        dt, sat = self.PathAugmflow(path)
        dt=dt
        for sedg in sat:
            p = sedg[0]
            q = sedg[1]
            if self.Tree[p] == self.Tree[q] and self.Tree[p] == self.TR_S:
                self.Parent[q] = -1
                # print("Creating orphan: ",q)
                self.Orphans.add(q)
            if self.Tree[p] == self.Tree[q] and self.Tree[p] == self.TR_T:
                self.Parent[p] = -1
                # print("Creating orphan: ",p)
                self.Orphans.add(p)
        # print("Orphans in end augment stage: ",self.Orphans)

    def getValidParent(self, p):
        neigh_p = self.Neighbours[p]
        for q in neigh_p:
            orig_q = self.trackOrigin(q)

            if self.Tree[p] == self.Tree[q] and self.tree_cap_inv(q, p) > 0 and (
                    orig_q == self.SOURCE or orig_q == self.TAP):
                return q
        return None

    def processOrphan(self, p):

        par_p = self.getValidParent(p)
        if par_p is not None:
            self.Parent[p] = par_p
        else:
            neigh_p = self.Neighbours[p]
            for q in neigh_p:
                if self.Tree[p] == self.Tree[q]:
                    if self.tree_cap_inv(q, p) > 0:
                        self.Actq.append(q)
                        self.Actives[q] = True
                    if self.Parent[q] == p:
                        self.Orphans.add(q)
                        self.Parent[q] = -1
            self.Tree[p] = self.TR_NULL

            self.Actives[p] = False
            return None

    def adoption_stage(self):
        while len(self.Orphans) != 0:
            p = self.Orphans.pop()
            self.processOrphan(p)

    def Kolmogorov(self):
        print("Ruunning Kolmogorov solver")
        while True:
            # print("##########################################")
            path = self.grow_stage()
            #print("Path: ", path)
            if path == None:
                break

            self.augment_stage(path)

            self.adoption_stage()
            self.resFlow=self.Graph-self.Flow
            self.resFlow=self.resFlow


        for key, value in enumerate(self.Tree):
            if value == self.TR_S:
                # if value==self.TR_S or value==self.TR_NULL:
                self.source_group.append(key)
            else:
                self.tap_group.append(key)
        for vs in self.source_group:
            for vt in self.tap_group:
                if (self.Graph[vs][vt] > 0 and (self.Graph[vs][vt] - self.Flow[vs][vt]) == 0) or (
                        self.Graph[vt][vs] > 0 and (self.Graph[vt][vs] - self.Flow[vt][vs]) == 0):
                    self.cuts.append((vs, vt))
        self.maxflow = np.sum(self.Flow[self.SOURCE, :])


if __name__ == "__main__":
    Nvertex = 4
    Graph = [[0 for i in range(Nvertex)] for j in range(Nvertex)]
    Graph[0][1] = 9
    Graph[0][2] = 2
    Graph[1][3] = 4
    Graph[2][3] = 5
    Graph[1][2] = 5
    Graph[2][1] = 0
    # Graph[4][0]=10

    # Graph[3][1]=10
    # Graph[3][2]=10

    solver = KolmogorovSolver(Graph, 0, 3)
    solver.Kolmogorov()
    # solver.Tree
    # solver.tap_group
    print(solver.cuts)