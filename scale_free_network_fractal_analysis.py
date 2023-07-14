#!/usr/bin/env python
# coding: utf-8

# In[9]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import powerlaw
import os

from scipy import stats
from scipy.stats import t
from pylab import *
from collections import Counter
from itertools import count
        
get_ipython().run_line_magic('matplotlib', 'inline')


class SoneiraPeebles:
    """ Uma classe para gerar coordenadas de pontos com uma dimensão fractal especifica.
       
        Atributos
        ----------
        
        x0 : float
            Um número real para a origem da abcissa
        y0 : float
            Um número real para a origem da ordenada
        R : int or float
            O raio do círculo com origem em (x0, y0)
        eta : número inicial de pontos a serem gerados

        Métodos
        --------

        first_level()
            Gera as coordenadas dos pontos do primeiro nível e retorna uma mensagem de confirmação.

        levels(xs, ys, lamb, eta)
            Gera as coordenadas do nível subsequente e retorna uma mensagem de confirmação.

        points()
            Retorna duas listas uma para as coordenadas da abicissa e outra para a coordenada
            das ordenadas.
        
        """
    def __init__(self, x0, y0, R, eta):
        self.x0 = x0
        self.y0 = y0
        self.R = R
        self.eta = eta
        self.x = []
        self.y = []
        
    def first_level(self):
        
        def generate_points(xi, yi, R):
            r = uniform(0, R)
            theta = random()*2*np.pi
            u = xi + r*np.cos(theta) + random()*np.cos(theta)
            v = yi + r*np.sin(theta) + random()*np.sin(theta)
            return u, v
        
        for i in range(self.eta):
            cx, cy = generate_points(self.x0, self.y0, self.R)
            self.x.append(cx)
            self.y.append(cy)
        
        return "Points generated!"    
            
    def levels(self, xs, ys, lamb, eta):
        
        def generate_points(xi, yi, R):
            r = uniform(0, R)
            theta = random()*2*np.pi
            u = xi + r*np.cos(theta) + random()*np.cos(theta)
            v = yi + r*np.sin(theta) + random()*np.sin(theta)
            return u, v
        
        r = self.R/lamb
    
        for i in range(len(xs)):
            x0 = xs[i]
            y0 = ys[i]
            for j in range(eta):
                cx, cy = generate_points(x0, y0, r)
                self.x.append(cx)
                self.y.append(cy)
        
        return "Points generated!"
    
    def points(self):
        return self.x, self.y


class FractalDimension:
    """ Uma classe para calcular a dimensão fractal de uma conjunto de pontos com 
        coordenadas normalizada entre 0 e 1 usando o método box-counting.
        
        Atributos
        ----------
        
        coords : tuple
            Uma tupla com para os pares ordenados para abcissa e a ordenada dos pontos
        scales : list
            Uma lista com as escalas das caixa para calcular a dimensão fractal, 
            por exemplo "scales = [0.001, 0.002, ...]". As escalas devem ser construidas
            no intervalo 0 < x, y < 1. 
        box : tuple
            Uma tupla com as dimensões das caixas, por exemplo: (xmin, xmax, ymin, ymax).
            
        Métodos
        --------

        show()
            Plota os pontos do fractal.

        dimension(xs, ys, lamb, eta)
            Estima a dimensão fractal usando o método box-counting.

        best_fit()
            Retorna o melhor fit dos dados.
            
        statistics()
            Retorna duas listas uma para as coordenadas da abicissa e outra para a coordenada
            das ordenadas.
        
        histogram()
            Retorna os valores da inclinação e da intersecção e suas incertezas com 95% de confiança.
        
        """
    def __init__(self, coords, scales, box):
        self.coords = coords
        self.scales = scales
        self.box = box
        self.dim = []
        self.boxes = []
        
    def show(self):
        
        x = []
        y = []
        for i in range(len(self.coords)):
            x.append(self.coords[i][0])
            y.append(self.coords[i][1])
        
        plt.figure(figsize=(5,5))
        plt.plot(x, y, 'o', c='k', markersize=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        
        return plt.show()
    
        
    def dimension(self):
        
        def box_counting(coords, l, box):
        
            xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
    
            n = 1/l
    
            d = (xmax - xmin)/n
    
            u = np.arange(xmin, xmax + d, d)
            v = np.arange(ymin, ymax + d, d)
    
            q = lambda x, y: [(x, y), (x+d, y), (x+d, y+d), (x, y+d)]
            sq = [[q(x, y) for x in u[:-1]] for y in v[:-1]] 
    
            cont = 0
            for i in range(len(sq)):
                for j in range(len(sq)):
                    a, b, c, d = sq[i][j]
                    for k in range(len(coords)):
                        px, py = coords[k]
                        teste = ((px>a[0])&(px<b[0]))&((py>b[1])&(py<d[1]))
                        if teste == True:
                            cont += 1
                            break
            return cont
        
        boxes_number = [box_counting(self.coords, l, self.box) for l in self.scales]
        self.boxes.append(boxes_number)
    
        res = stats.linregress(np.log(self.scales), np.log(boxes_number))
        self.dim.append((res.slope, res.intercept, res.rvalue**2))
        
        return print('Fractal dimension: {}'.format(round(-res.slope,2)))
    
    def best_fit(self):
        
        def power_law(x, a, b):
            return b*x**a
        
        xteo = np.linspace(np.min(self.scales), np.max(self.scales), len(self.scales))    
        yteo = [power_law(x, self.dim[0][0], np.exp(self.dim[0][1])) for x in xteo]
        
        plt.figure(figsize=(5,5))
        plt.plot(self.scales, self.boxes[0], '-', c='tab:green', label='data')
        plt.plot(xteo, yteo, '--', c='k', label='best fit = %.2f, $R^{2}$ = %.2f'%(self.dim[0][0], round(self.dim[0][2],2)))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('size')
        plt.ylabel('boxes')
        plt.legend(fontsize=10)
        
        return plt.show()
    
    def statistics(self):
        
        res = stats.linregress(np.log(self.scales), np.log(self.boxes[0]))
        tinv = lambda p, df: abs(t.ppf(p/2, df))
        ts = tinv(0.05, len(self.scales)-2)
        
        print(f"slope (95%): {res.slope:.6f} +/- {ts*res.stderr:.6f}",
              f"intercept (95%): {res.intercept:.6f} +/- {ts*res.intercept_stderr:.6f}", sep=os.linesep)
        
        
    def histogram(self):
        
        n = len(self.boxes[0])
        mean = np.mean(self.boxes[0])
        err = 1.64*np.std(self.boxes[0])/np.sqrt(n) #90% confidence 
        
        plt.figure(figsize=(5,5))
        plt.hist(self.boxes[0], bins=int(np.sqrt(n)), color='tab:green', label= 'mean: %.2f, std: %.2f'%(mean, err))
        plt.xlabel('boxes')
        plt.ylabel('frequency')
        plt.legend()
        
        return plt.show()



class ScaleFreeNetwork:
    """ Uma classe para geral uma rede livre de escala usando dois métodos: 1) a probabilidade 
        de ligação entre os nós e dado por "p(ki) = ki / sum kj" onde ki é o grau do nó i e sum kj é a soma
        de todos os graus da rede. 2) "1/ dij**(df)" one d é a distância euclidiana entre os nós i e j, enquanto
        que df é a dimensão fractal do fractal estimada pelo método box-counting.
       
        Atributos
        ----------
        
        coords : tuple
            Uma tupla com para os pares ordenados para abcissa e a ordenada dos pontos
        scales : list
            Uma lista com as escalas das caixa para calcular a dimensão fractal, 
            por exemplo "scales = [0.001, 0.002, ...]". As escalas devem ser construidas
            no intervalo 0 < x, y < 1. 
        box : tuple
            Uma tupla com as dimensões das caixas, por exemplo: (xmin, xmax, ymin, ymax).

        Métodos
        --------

        generate_network()
            Plota os pontos do fractal.

        connect_network_with_degree_probability(xs, ys, lamb, eta)
            Estima a dimensão fractal usando o método box-counting.
             
        connect_network_with_fractal_dimension()
            Estima a dimensão fractal usando o método box-counting.
             
        plot_scale_free_network()
            Plota os pontos do fractal.
             
        probability_distribution_data()
            calcula os graus e a probabilidade 

        fit_probability_distribution(degree, probability, var_x, var_y, figsize)
            Retorna o melhor fit dos dados.
        
        sigma_gamma(gamma, sigma_gamma, fd, sigma_fd)
            Retorna a incerteza da dimensão fractal de grau a partir dos valores: gamma (expoente de escala),
            sigma_gamma (incerteza padrão de gamma), df (dimensão fractal dos pontos obtidos 
            pelo método box-counting), sigma_df (incerteza padrão de df), usando propagração de erros a partir
            da fórmula "gamma = 1 + df/dk", onde dk é a dimensão fractal de grau.
            
        """
    
    def __init__(self, dict_nodes_pos):
        self.dict_nodes_pos = dict_nodes_pos
        
    def generate_network(self):
        
        # generate a network
        G = nx.Graph()
        
        # add nodes
        G.add_nodes_from(list(self.dict_nodes_pos.keys()))
        
        # assigning positions
        nx.set_node_attributes(G, self.dict_nodes_pos, 'pos')
        
        return G
    
    
        
    def connect_network_with_degree_probability(self, G, m, n):
        
        def generate_initial_edges(G, m):
    
            nos = list(G.nodes())

            if m == 0:
                return 'The number of starting edges (m) must be different from zero!'
            else:
                links = np.random.choice(nos, int(2*m))
                us = links[:m]
                vs = links[-m:]

                uv = list(zip(us, vs))

                G.add_edges_from(uv)

                return G
    
        H = generate_initial_edges(G, m)
        
        for _ in range(n):
            nodes = list(H.nodes())
            # Select a random node to connect
            new_node = np.random.choice(nodes)

            # Calculate the connection probability for each existing node
            # based on the relationship k_i /\sum k_j where k_i is the degree of node i
            # and \sum k_j is the sum of all degrees in the network
            probs = []
            for node in nodes:
                degree = H.degree(node)
                prob_connection = degree / sum(list(dict(H.degree()).values()))#d
                probs.append(prob_connection)

            # Select an existing node to connect based on probabilities
            select_node = np.random.choice(nodes, p=probs)

            # Add a new connection between the new node and the existing node
            H.add_edge(new_node, select_node)

        return H
    
    def connect_network_with_fractal_dimension(self, G, m, n, fd):
        
        def dist(G, u, v):
    
            x0, y0 =  G.nodes[u]['pos']
            x1, y1 =  G.nodes[v]['pos']

            d = ((x1-x0)**2+(y1-y0)**2)**0.5

            return d
        
        def generate_initial_edges(G, m):
    
            nos = list(G.nodes())

            if m == 0:
                return 'The number of starting edges (m) must be different from zero!'
            else:
                links = np.random.choice(nos, int(2*m))
                us = links[:m]
                vs = links[-m:]

                uv = list(zip(us, vs))

                G.add_edges_from(uv)

                return G
    
        H = generate_initial_edges(G, m)
        
        for _ in range(n):
            nodes = list(H.nodes())
            # Select a random node to connect
            new_node = np.random.choice(nodes)
    
            # Calculate the connection probability for each existing node
            # according to 1/d**df where d is the Euclidean distance between
            # the nodes and df is the fractal dimension of the points
            probs = []
            for node in nodes:
                d = dist(H, node, new_node)**fd
                if d != 0:
                    prob_connection = 1/d
                    probs.append(prob_connection)
                else:
                    probs.append(1)
                    
            # Select an existing node to connect based on probabilities
            select_node = np.random.choice(nodes, p=probs)

            # Add a new connection between the new node and the existing node
            H.add_edge(new_node, select_node)

        return H
    
    def plot_scale_free_network(self, G, dict_nodes_pos, figsize, node_size):
    
        # creating the dictionary of the degree centrality
        degree_dict = dict(G.degree())

        nodes = list(degree_dict.keys())

        nx.set_node_attributes(G, degree_dict, "degree")

        # creating the list of the degree centrality
        degree_list = list(degree_dict.values())

        # creating the degree frequency dictionary
        degree_frequency = Counter(degree_list)

        degree = list(degree_frequency.keys())

        degree_sorted = sorted(degree)

        groups_dg = set(degree_dict.values())

        mapping_dg = dict(zip(sorted(groups_dg), count()))

        colors_dg = [mapping_dg[G.nodes[n]["degree"]] for n in nodes]

        # ploting the network accourding to the Bonacich centrality 
        plt.figure(figsize=figsize)

        # drawing nodes and edges separately so we can capture collection for colobar
        ec_dg = nx.draw_networkx_edges(G, pos = dict_nodes_pos, alpha=1)
        nc_dg = nx.draw_networkx_nodes(G, pos = dict_nodes_pos, nodelist=nodes, node_color=colors_dg,
                                       node_size=[(v * node_size)+1 for v in degree_list], cmap=plt.cm.jet)
        plt.colorbar(nc_dg, fraction=0.02, pad=0.04)
        plt.axis('off')
        
        return plt.show()
    
    def probability_distribution_data(self, G):
        
        degree = list(dict(G.degree()).values())

        degree_set = list(set(degree))

        degree_cont = [degree.count(k) for k in degree_set]

        prob = [k/len(degree) for k in degree_cont]
        
        return degree, prob

    def fit_probability_distribution(self, degree, probability, var_x, var_y, figsize):

        def power_law(x, a, b):
            return b*x**a

        tinv = lambda p, dff: abs(t.ppf(p/2, dff))
        
        x, y = degree, probability

        xmin = powerlaw.Fit(y).xmin
        ind_xmin = y.index(xmin) 

        u,  v = x[:ind_xmin], y[:ind_xmin]

        res = stats.linregress(np.log(u), np.log(v))
        ts = tinv(0.05, len(u)-2)
        a, a_std, b, r2 = res.slope, ts*res.stderr, res.intercept, res.rvalue**2

        xteo = list(np.linspace(min(x), max(x), 100))
        yteo = [power_law(x, a, np.exp(b)) for x in xteo]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(x, y, 'o', c='tab:blue', markersize=5, fillstyle='none', label='data')
        ax.plot(xteo, yteo, '-', c='k', linewidth=1, label = 'slope = %.2f $\pm$ %.2f, $R^{2} = %.2f$'%(a, a_std, r2), alpha=1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        plt.legend(fontsize=8)

        return plt.show()
    
    def sigma_gamma(self, gamma, sigma_gamma, fd, sigma_fd):
        
        return (1/((gamma-1)**2))*(np.sqrt(((gamma-1)**2)*sigma_fd**2+(fd**2)*(sigma_gamma**2)))


# In[ ]:



