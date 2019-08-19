from __future__ import division
import matplotlib.image as mpimg
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
import random, math
from scipy.spatial import Voronoi, voronoi_plot_2d, Delaunay
from Bio import SeqIO, Seq
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import pytess
import networkx as nx
import os, datetime
import planarity
import ipdb
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import matplotlib
import matplotlib.cm as cm
import scipy.optimize
import warnings
import itertools
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial import cKDTree
from sklearn import preprocessing
import collections
try:
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
except ImportError:
    raise ImportError("Matplotlib is required for draw()")





class Surface:
    def __init__(self, target_nsites, directory_label, random_seed = 4, master_directory = None):
        self.target_nsites = target_nsites
        self.directory_label = directory_label
        random.seed(a=random_seed)
        today = datetime.date.today()
        if not master_directory:
            self.directory_name = str(today) + '_' + str(self.directory_label) + '_' + str(time.time())
            os.makedirs(self.directory_name)
        else:
            self.directory_name = master_directory

    def circular_grid_ian(self):
        self.adjusted_target_nsites = int(self.target_nsites*(0.866193598020831))
        self.roi_radius = np.sqrt((self.adjusted_target_nsites) / np.pi)
        self.grid_radius = int(np.ceil(self.roi_radius))
        self.box_dim = 2 * self.grid_radius

        self.box_half_nsites = int(np.ceil((2*self.grid_radius) ** 2))
        self.lattice_sites_a = np.zeros((self.box_half_nsites, 2))
        self.lattice_sites_b = np.zeros((self.box_half_nsites, 2))
        self.all_neighbor_sets = [[]]*self.box_half_nsites*2
        # self.a_neighbor_sets = [[]]*self.box_half_nsites
        # self.b_neighbor_sets = [[]] * self.box_half_nsites
        self.gridID_catalog = {}

        for i in range(0, self.box_dim):
            for jj in range(0, self.box_dim):
                current_site_identity = i * self.box_dim + jj
                b_site_parallel_identity = current_site_identity + self.box_half_nsites
                self.lattice_sites_a[current_site_identity, 0] = float(i) - float(self.roi_radius)
                self.lattice_sites_a[current_site_identity, 1] = (float(jj) * math.sqrt(3)) - float(self.roi_radius)
                a_site_neighbors = []
                b_site_neighbors = []



                # try:
                a_above = current_site_identity - self.box_dim
                if a_above < 0: #upper edge boundary
                    pass
                else:
                    a_site_neighbors.append(a_above)

                a_below = current_site_identity + self.box_dim
                if a_below >= self.box_half_nsites: #lower edge boundary
                    pass
                else:
                    a_site_neighbors.append(a_below)

                a_top_left = current_site_identity - 1 - self.box_dim + self.box_half_nsites
                if current_site_identity%self.box_dim - (current_site_identity - 1)%self.box_dim != 1 or current_site_identity - self.box_dim < 0:#reject
                    pass
                else:
                    a_site_neighbors.append(a_top_left)

                a_top_right = current_site_identity - self.box_dim + self.box_half_nsites
                if current_site_identity - self.box_dim < 0 :#upper criterion
                    pass
                else:
                    a_site_neighbors.append(a_top_right)
                # ipdb.set_trace()

                a_lower_left = current_site_identity - 1 + self.box_half_nsites
                if current_site_identity%self.box_dim - (current_site_identity - 1)%self.box_dim != 1:#lower criterion, left edge criterion
                    pass
                else:
                    a_site_neighbors.append(a_lower_left)

                a_lower_right = current_site_identity + self.box_half_nsites
                a_site_neighbors.append(a_lower_right)


                b_above = current_site_identity + self.box_half_nsites - self.box_dim
                if b_above - self.box_half_nsites < 0: #upper edge boundary
                    pass
                else:
                    b_site_neighbors.append(b_above)

                b_below = current_site_identity + self.box_half_nsites + self.box_dim
                if b_below >= self.box_half_nsites*2: #upper edge boundary
                    pass
                else:
                    b_site_neighbors.append(b_below)

                b_top_right = current_site_identity + 1
                if current_site_identity%self.box_dim - (current_site_identity + 1)%self.box_dim != -1 :#upper criterion
                    pass
                else:
                    b_site_neighbors.append(b_top_right)

                b_top_left = current_site_identity
                b_site_neighbors.append(b_top_left)

                b_lower_left = current_site_identity + self.box_dim
                if b_lower_left >= self.box_half_nsites:#lower criterion, left edge criterion
                    pass
                else:
                    b_site_neighbors.append(b_lower_left)

                b_lower_right = current_site_identity + self.box_dim + 1
                if b_lower_right >= self.box_half_nsites or current_site_identity%self.box_dim - (current_site_identity + 1)%self.box_dim != -1:
                    pass
                else:
                    b_site_neighbors.append(b_lower_right)

                self.all_neighbor_sets[current_site_identity] = a_site_neighbors
                self.all_neighbor_sets[b_site_parallel_identity] = b_site_neighbors
        # all_neighbors = np.append(a_site_neighbors,b_site_neighbors)
        self.lattice_sites_b[:, 0] = self.lattice_sites_a[:, 0] + 0.5
        self.lattice_sites_b[:, 1] = self.lattice_sites_a[:, 1] + math.sqrt(3) / 2

        self.lattice_sites_both = np.append(self.lattice_sites_a, self.lattice_sites_b, axis=0)
        self.site_coordinates = np.zeros((0, 2))
        self.old_site_IDs = []
        self.final_neighborsets = []
        #circle criterion
        new_site_counter = 0
        for oldID in range(0, len(self.lattice_sites_both)):
            # if point[0]**2 + point[1]**2 <= self.roi_radius ** 2:  # then the point is inside the circle
            if np.linalg.norm(self.lattice_sites_both[oldID]) <= self.roi_radius:
                self.site_coordinates = np.append(self.site_coordinates, [self.lattice_sites_both[oldID]], axis=0)
                self.old_site_IDs.append(oldID)
                # ipdb.set_trace()
                self.gridID_catalog.update({oldID:new_site_counter})

                new_site_counter += 1
        for point in self.old_site_IDs:
            # ipdb.set_trace()
            accepted_neighbors = [self.gridID_catalog[oldID] for oldID in list(set(self.all_neighbor_sets[point]).intersection(self.old_site_IDs))]
                # [self.gridID_catalog[i] for i in self.all_neighbor_sets[point] and self.old_site_IDs]
            self.final_neighborsets.append(accepted_neighbors)
        # ipdb.set_trace()
        self.nsites = len(self.site_coordinates)
        pickasite = 1
        # self.all_neighbor_sets[pickasite] = a_site_neighbors
        # self.all_neighbor_sets[pickasite + self.box_half_nsites] = b_site_neighbors
        # plt.scatter(self.lattice_sites_both[:,0],self.lattice_sites_both[:,1])
        # plt.scatter(self.lattice_sites_both[pickasite][0], self.lattice_sites_both[pickasite][1], c='red')
        # for neighbor in self.all_neighbor_sets[pickasite]:
        #     plt.scatter(self.lattice_sites_both[neighbor][0],self.lattice_sites_both[neighbor][1],c='yellow')
        # plt.show()
        # ipdb.set_trace()
        # plt.scatter(self.site_coordinates[:,0],self.site_coordinates[:,1])
        # plt.scatter(self.site_coordinates[pickasite][0], self.site_coordinates[pickasite][1], c='red')
        # for neighbor in self.final_neighborsets[pickasite]:
        #     plt.scatter(self.site_coordinates[neighbor][0],self.site_coordinates[neighbor][1],c='yellow')
        # plt.show()
        # ipdb.set_trace()
        # merge = np.vstack((lattice_sites_a, lattice_sites_b))

    def plot_sites(self):
        plt.scatter(self.site_coordinates[:, 0], self.site_coordinates[:, 1])
        plt.show()



    def scale_image_axes(self, filename = 'monalisacolor.png'):
        #for resolving aspect ratio and scale differences of the image and the grid
        imarray = np.array(mpimg.imread(filename))
        x_length_image = len(imarray[0,:])
        y_length_image = len(imarray[:,0])
        x_grid_min = min(self.site_coordinates[:,0])
        y_grid_min = min(self.site_coordinates[:, 1])
        x_grid_max = max(self.site_coordinates[:,0])
        y_grid_max = max(self.site_coordinates[:,1])
        x_length_grid = x_grid_max-x_grid_min
        y_length_grid = y_grid_max-y_grid_min
        aspect_ratio_image = x_length_image/y_length_image
        aspect_ratio_grid = x_length_grid/y_length_grid
        if aspect_ratio_image <= aspect_ratio_grid: #then this means that the image is tall relative to the grid
            # in this case, we add padding to the sides of the image while maintaining the original height
            padding_one_side = np.ones((len(imarray[:,0]),int((1./aspect_ratio_grid*len(imarray[:,0])-len(imarray[0,:]))/2) ,np.shape(imarray)[2]))
            try :self.scaled_imarray = np.concatenate((padding_one_side,imarray),axis=1)
            except: ipdb.set_trace()
            self.scaled_imarray = np.concatenate((self.scaled_imarray, padding_one_side), axis=1)/np.max(imarray)
        elif aspect_ratio_image > aspect_ratio_grid:
            # in this case, we add padding to the top and bottom with same width as the original image
            padding_one_side = np.ones(( int((aspect_ratio_grid*len(imarray[0,:]) - len(imarray[:,0]))/2) , len(imarray[0,:]) , 3))
            self.scaled_imarray = np.concatenate((padding_one_side, imarray), axis=0)
            self.scaled_imarray = np.concatenate((self.scaled_imarray, padding_one_side), axis=0) / np.max(imarray)
        else:
            print 'error with aspect ratio scaling of image'
        self.lookup_x_axis = np.arange(x_grid_min,x_grid_max,x_length_grid/float(len(self.scaled_imarray[:,0])-1))
        self.lookup_y_axis = np.arange(y_grid_min,y_grid_max,y_length_grid/float(len(self.scaled_imarray[0,:])-1))



    def seed(self,nseed,full_output=True):
        self.nseed = nseed
        choices = [i for i in range(0, self.nsites)]
        self.seed = np.random.choice(choices, nseed, replace=False)
        self.corseed = np.zeros((nseed, 2))
        self.p1 = list()
        for i in range(0, nseed):
            n = self.seed[i]
            self.corseed[i] = self.site_coordinates[n, :]
            if full_output==True:
                np.savetxt(self.directory_name +'/'+ 'corseed', self.corseed, delimiter=",")
                np.savetxt(self.directory_name +'/'+  'seed', self.seed, delimiter=",")

        self.bc = ["" for x in range(0, nseed)]
        for i in range(0, nseed):

            self.bc[i] = barcode()
            if i >= 0:
                for j in range(0, i - 1):
                    while self.bc[i] == self.bc[j]:
                        self.bc[i] = barcode()

        bc_record = ["" for i in self.bc]
        for i in range(0, len(self.bc)):
            bc_record[i] = SeqRecord(Seq(self.bc[i]), id=str(i))

        for i in range(0, nseed):
            self.p1.append(str(bc_record[i].seq.reverse_complement()))
        if full_output == True:
            SeqIO.write(bc_record, self.directory_name + '/' + "bc.faa", "fasta")
            np.savetxt(self.directory_name + '/' + 'barcode', self.bc, delimiter=",", fmt="%s")
            np.savetxt(self.directory_name+'/'+ 'p1', self.p1, delimiter=',', fmt='%s')

        # return corseed, p1, bc, seed
    def ideal_only_crosslink(self,full_output=True):
        self.triangulated_delaunay = Delaunay(self.corseed, qhull_options="Qz Q2")
        self.ideal_graph = get_ideal_graph(self.triangulated_delaunay)
        self.rgb_stamp_catalog = np.zeros((self.nseed,3))
        voronoi_kdtree = cKDTree(self.corseed)
        test_point_dist, self.p2_proxm = voronoi_kdtree.query(self.site_coordinates)
        self.polony_sizes = collections.Counter(self.p2_proxm)
        # ipdb.set_trace()
        # self.p2_proxm = np.zeros((self.nsites))
        for i in range(0, self.nseed):
            c = random.random()
            for j in range(0, self.nsites):
                if self.p2_proxm[j] == i:
                    rgb, gray = get_pixel_rgb(self.scaled_imarray,self.lookup_x_axis,self.lookup_y_axis,self.site_coordinates[j,0],self.site_coordinates[j,1])
                    rgb = rgb / np.sum(rgb)
                    p_threshold = random.random()
                    if p_threshold < gray: #darker means closer to zero,
                        try:
                            colorID = np.random.choice([0,1,2],p=rgb)
                            self.rgb_stamp_catalog[i][colorID] += 1.
                        except:
                            self.rgb_stamp_catalog[i][0] += 1
                            self.rgb_stamp_catalog[i][1] += 1
                            self.rgb_stamp_catalog[i][2] += 1
            self.rgb_stamp_catalog[i] = self.rgb_stamp_catalog[i]/float(self.polony_sizes[i])
            # ipdb.set_trace()

    def crosslink_polonies(self, full_output=True, optimal_linking=True,bipartite=False):
        if optimal_linking == False:
            if full_output == True:
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)

            # nseed = len(p1)
            self.p2_proxm = np.zeros((self.nsites),dtype=int)
            self.p2_secondary = np.zeros((self.nsites),dtype=int)
            # self.sites_distance_matrix = np.zeros((self.nsites,self.nsites),dtype=np.float)
            self.sites_nearest_neighbors = np.zeros((self.nsites, 6), dtype=np.float)
            self.available_nearest_neighbors = []
            self.sites_all_partners = np.zeros((self.nsites, 2), dtype=np.float)
            self.site_membership = {}
            self.bc_index_pair_sitelinking = []
            self.self_pairing_events = []
            self.cross_pairing_events = []
            self.self_pairing_events_sites = []
            self.cross_pairing_events_sites = []
            #SITE MAPPING LOOP - gather information about site memborship and neighborhood
            if full_output == True:print 'begin gathering neighborhood data of sites'
            for i in range(0, self.nsites):
                self.p2_dis = np.zeros((self.nseed))
                for j in range(0, self.nseed):
                    # self.p2_dis[j] = (math.sqrt((self.site_coordinates[i, 0] - self.corseed[j, 0]) ** 2 + (self.site_coordinates[i, 1] - self.corseed[j, 1]) ** 2))
                    self.p2_dis[j] = np.linalg.norm(self.site_coordinates[i] - self.corseed[j])
                distances_sorted = np.argsort(self.p2_dis)
                self.p2_proxm[i] = distances_sorted[0] #get the membership in polony by taking the closest of distances to each seed
                self.p2_secondary[i] = distances_sorted[1] #get the second closest
                self.site_membership.update({i: self.p2_proxm[i]})
                # for j in range(0, self.nsites):
                #     self.sites_distance_matrix[i, j] = np.linalg.norm(self.site_coordinates[i] - self.site_coordinates[j])
                # self.sites_nearest_neighbors[i] = np.argsort(self.sites_distance_matrix[i])[1:7]
                # self.available_nearest_neighbors.append(list(self.sites_nearest_neighbors[i]))
                self.available_nearest_neighbors.append(list(self.final_neighborsets[i]))
            #crosslinking step - pick a random nearest neighbor, update partnership, keep track of self pairing and crosspairing
            self.sites_nearest_neighbors = self.final_neighborsets
            if full_output == True:print 'begin linking sites'
            visitation_queue = range(0,self.nsites)
            while len(visitation_queue) >= 2:
                arbitrary_site = visitation_queue[0]
                if len(self.available_nearest_neighbors[arbitrary_site]) != 0:
                    partner = random.choice(self.available_nearest_neighbors[arbitrary_site])
                    if partner in visitation_queue:
                        try:
                            for neighbor in self.sites_nearest_neighbors[int(partner)]:
                                try:
                                    self.available_nearest_neighbors[int(neighbor)].remove(partner)
                                except:
                                    pass
                        except:
                            pass
                        nascent_pair_event = [self.site_membership[arbitrary_site],self.site_membership[partner]]
                        nascent_pair_event_sites = [arbitrary_site, partner]
                        self.bc_index_pair_sitelinking.append(nascent_pair_event)
                        if nascent_pair_event[0] == nascent_pair_event[1]:
                            self.self_pairing_events.append(nascent_pair_event)
                            self.self_pairing_events_sites.append(nascent_pair_event_sites)
                        else:
                            self.cross_pairing_events.append(nascent_pair_event)
                            self.cross_pairing_events_sites.append(nascent_pair_event_sites)
                        visitation_queue.remove(arbitrary_site)
                        visitation_queue.remove(partner)
                    else: #the selected partner is not available, has already been occupied, we should therefore remove it as an option all over
                        self.available_nearest_neighbors[arbitrary_site].remove(int(partner))
                        for neighbor in self.sites_nearest_neighbors[int(partner)]:
                            try:
                                self.available_nearest_neighbors[int(neighbor)].remove(partner)
                            except:
                                pass
                else:
                    visitation_queue.remove(arbitrary_site)

            # plt.scatter(self.site_coordinates[:,0],self.site_coordinates[:,1],color='black')

            # plt.scatter(self.site_coordinates[pickasite][0], self.site_coordinates[pickasite][1], c='red')
            # for neighbor in self.final_neighborsets[pickasite]:
            #     plt.scatter(self.site_coordinates[neighbor][0],self.site_coordinates[neighbor][1],c='yellow')
            # plt.show()
            # ipdb.set_trace()

            # ipdb.set_trace()
            if full_output == True: print 'end of site linking'
            self.p2 = ["" for x in range(0, self.nsites)]
            for i in range(0, self.nsites):
                x = int(self.p2_proxm[i])
                self.p2[i] = reverse(self.p1[x])

            #self.bc_index_pair = zip(self.p2_proxm, self.p2_secondary) #create crosslinks, for each site, connect it to its parent and closest polony neighbor
            self.bc_index_pair = self.cross_pairing_events
            # ipdb.set_trace()
            cmap = matplotlib.cm.get_cmap('Spectral')
            vor_polony = Voronoi(self.corseed, qhull_options = "")
            if full_output == True:
                if self.nseed <= 500:
                    voronoi_plot_2d(vor_polony,show_vertices=False)
                else:
                    voronoi_plot_2d(vor_polony, show_vertices=False,show_points = False, line_width = 0.1)
            self.rgb_stamp_catalog = np.zeros((self.nseed,3))
            if full_output == True:print 'begin gathering polony sizes'
            self.polony_sizes = collections.Counter(self.p2_proxm) #p2_proxm stores which seed each site is closest to
            # ipdb.set_trace()
            if full_output == True:print 'end of gathering polony sizes'
            for i in range(0, self.nseed):
                c = random.random()
                for j in range(0, self.nsites):
                    if self.p2_proxm[j] == i:
                        if full_output == True:
                            plt.scatter(self.site_coordinates[j, 0], self.site_coordinates[j, 1], color=cmap(c), s=1)
                        rgb, gray = get_pixel_rgb(self.scaled_imarray,self.lookup_x_axis,self.lookup_y_axis,self.site_coordinates[j,0],self.site_coordinates[j,1])
                        rgb = rgb / np.sum(rgb)
                        p_threshold = random.random()
                        if p_threshold < gray: #darker means closer to zero,
                            try:
                                colorID = np.random.choice([0,1,2],p=rgb)
                                self.rgb_stamp_catalog[i][colorID] += 1.
                            except:
                                self.rgb_stamp_catalog[i][0] += 1
                                self.rgb_stamp_catalog[i][1] += 1
                                self.rgb_stamp_catalog[i][2] += 1
                self.rgb_stamp_catalog[i] = self.rgb_stamp_catalog[i]/float(self.polony_sizes[i])
                # ipdb.set_trace()
            if full_output == True:print 'end of stamp cataloging'
            self.triangulated_delaunay = Delaunay(self.corseed, qhull_options="Qz Q2")

            self.ideal_graph = get_ideal_graph(self.triangulated_delaunay)

            self.untethered_graph = nx.Graph()
            for edge in self.cross_pairing_events:
                self.untethered_graph.add_edge(edge[0],edge[1])
                self.untethered_graph.add_edge(edge[1], edge[0])

            if full_output == True:
                for edge in self.cross_pairing_events_sites:
                    plt.plot([self.site_coordinates[edge[0]][0], self.site_coordinates[edge[1]][0]],
                             [self.site_coordinates[edge[0]][1], self.site_coordinates[edge[1]][1]], color='red')
                for edge in self.self_pairing_events_sites:
                    plt.plot([self.site_coordinates[edge[0]][0], self.site_coordinates[edge[1]][0]],
                             [self.site_coordinates[edge[0]][1], self.site_coordinates[edge[1]][1]], color='gray')
                np.savetxt(self.directory_name + '/' + 'p2', self.p2, delimiter=",", fmt="%s")
                np.savetxt(self.directory_name + '/' + 'polony_seed', self.p2_proxm, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'point_adjacent_polony', self.p2_secondary, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'bcindexpairlist', self.bc_index_pair, delimiter=',')
                # plt.rcParams['figure.figsize'] = (20,20)
                plt.xlim(-.1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.min(
                    self.site_coordinates[:, 0]),
                         .1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.max(
                             self.site_coordinates[:, 0]))
                plt.ylim(-.1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.min(
                    self.site_coordinates[:, 1]),
                         .1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.max(
                             self.site_coordinates[:, 1]))
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=0.5)

                # plt.figure(figsize=(10, 10), dpi=500)
                else:
                    plt.autoscale(enable=True, axis='both', tight=None)
                plt.savefig(self.directory_name+'/'+'polony.svg')
                plt.savefig(self.directory_name + '/' + 'polony.png')
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)
                plt.autoscale(enable=True, axis='both', tight=None)
                plt.triplot(self.corseed[:,0],self.corseed[:,1],self.triangulated_delaunay.simplices.copy(),linestyle='solid',alpha=1,marker=None,color='#3498DB')
                if self.nseed <= 500:
                    plt.plot(self.corseed[:,0],self.corseed[:,1],'o',color='#3498DB',markerfacecolor='w',markersize=12)
                # plt.rcParams['figure.figsize'] = (20, 20)
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=1, fontsize=6,horizontalalignment='center', verticalalignment='center')
                plt.savefig(self.directory_name + '/' +'delaunay_polony.svg')
                plt.savefig(self.directory_name + '/' + 'delaunay_polony.png')
                plt.close()
                plt.autoscale(enable=False, axis='both', tight=None)

        elif optimal_linking == True and bipartite == False:
            if full_output == True:
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)

            # nseed = len(p1)
            self.p2_proxm = np.zeros((self.nsites))
            self.p2_secondary = np.zeros((self.nsites))
            for i in range(0, self.nsites):
                self.p2_dis = np.zeros((self.nseed))
                for j in range(0, self.nseed):
                    # self.p2_dis[j] = (math.sqrt((self.site_coordinates[i, 0] - self.corseed[j, 0]) ** 2 + (self.site_coordinates[i, 1] - self.corseed[j, 1]) ** 2))
                    self.p2_dis[j] = np.linalg.norm(self.site_coordinates[i] - self.corseed[j])
                self.p2_proxm[i] = np.argsort(self.p2_dis)[0] #get the membership in polony by taking the closest of distances to each seed
                self.p2_secondary[i] = np.argsort(self.p2_dis)[1] #get the second closest

            self.p2 = ["" for x in range(0, self.nsites)]
            for i in range(0, self.nsites):
                x = int(self.p2_proxm[i])
                self.p2[i] = reverse(self.p1[x])

            self.bc_index_pair = zip(self.p2_proxm, self.p2_secondary)

            # ipdb.set_trace()
            cmap = matplotlib.cm.get_cmap('Spectral')
            vor_polony = Voronoi(self.corseed, qhull_options = "")
            if full_output == True:
                if self.nseed <= 500:
                    voronoi_plot_2d(vor_polony,show_vertices=False)
                else:
                    voronoi_plot_2d(vor_polony, show_vertices=False,show_points = False, line_width = 0.1)
            self.rgb_stamp_catalog = np.zeros((self.nseed,3))
            if full_output == True:print 'begin gathering polony sizes'
            self.polony_sizes = collections.Counter(self.p2_proxm) #p2_proxm stores which seed each site is closest to
            # ipdb.set_trace()
            if full_output == True:print 'end of gathering polony sizes'
            for i in range(0, self.nseed):
                c = random.random()
                for j in range(0, self.nsites):
                    if self.p2_proxm[j] == i:
                        if full_output == True:
                            plt.scatter(self.site_coordinates[j, 0], self.site_coordinates[j, 1], color=cmap(c), s=1)
                        rgb, gray = get_pixel_rgb(self.scaled_imarray,self.lookup_x_axis,self.lookup_y_axis,self.site_coordinates[j,0],self.site_coordinates[j,1])
                        rgb = rgb / np.sum(rgb)
                        p_threshold = random.random()
                        if p_threshold < gray: #darker means closer to zero,
                            try:
                                colorID = np.random.choice([0,1,2],p=rgb)
                                self.rgb_stamp_catalog[i][colorID] += 1.
                            except:
                                self.rgb_stamp_catalog[i][0] += 1
                                self.rgb_stamp_catalog[i][1] += 1
                                self.rgb_stamp_catalog[i][2] += 1
                self.rgb_stamp_catalog[i] = self.rgb_stamp_catalog[i]/float(self.polony_sizes[i])
                # ipdb.set_trace()
            if full_output == True:print 'end of stamp cataloging'
            self.triangulated_delaunay = Delaunay(self.corseed, qhull_options="Qz Q2")

            self.ideal_graph = get_ideal_graph(self.triangulated_delaunay)

            ajc_polony = np.zeros((len(self.bc), len(self.bc)))
            ajc_polony.astype(int)
            for i in range(0, len(self.bc_index_pair)):
                rrr = int(self.bc_index_pair[i][0])
                ccc = int(self.bc_index_pair[i][1])
                ajc_polony[rrr, ccc] = 1
                ajc_polony[ccc, rrr] = 1

            if full_output==True: np.savetxt(self.directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")

            self.untethered_graph = nx.Graph()
            for i in range(0, ajc_polony.shape[0]):
                self.untethered_graph.add_node(i)
                for j in range(0, ajc_polony.shape[1]):
                    if ajc_polony[i][j] == 1:
                        self.untethered_graph.add_edge(i, j)


            if full_output == True:
                np.savetxt(self.directory_name + '/' + 'p2', self.p2, delimiter=",", fmt="%s")
                np.savetxt(self.directory_name + '/' + 'polony_seed', self.p2_proxm, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'point_adjacent_polony', self.p2_secondary, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'bcindexpairlist', self.bc_index_pair, delimiter=',')
                # plt.rcParams['figure.figsize'] = (20,20)
                plt.xlim(-.1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.min(
                    self.site_coordinates[:, 0]),
                         .1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.max(
                             self.site_coordinates[:, 0]))
                plt.ylim(-.1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.min(
                    self.site_coordinates[:, 1]),
                         .1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.max(
                             self.site_coordinates[:, 1]))
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=0.5)

                # plt.figure(figsize=(10, 10), dpi=500)
                else:
                    plt.autoscale(enable=True, axis='both', tight=None)
                plt.savefig(self.directory_name+'/'+'polony.svg')
                plt.savefig(self.directory_name + '/' + 'polony.png')
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)
                plt.autoscale(enable=True, axis='both', tight=None)
                plt.triplot(self.corseed[:,0],self.corseed[:,1],self.triangulated_delaunay.simplices.copy(),linestyle='solid',alpha=1,marker=None,color='#3498DB')
                if self.nseed <= 500:
                    plt.plot(self.corseed[:,0],self.corseed[:,1],'o',color='#3498DB',markerfacecolor='w',markersize=12)
                # plt.rcParams['figure.figsize'] = (20, 20)
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=1, fontsize=6,horizontalalignment='center', verticalalignment='center')
                plt.savefig(self.directory_name + '/' +'delaunay_polony.svg')
                plt.savefig(self.directory_name + '/' + 'delaunay_polony.png')
                plt.close()
                plt.autoscale(enable=False, axis='both', tight=None)
            print 'end of crosslink polonies'

        elif optimal_linking == True and bipartite == True:
            if full_output == True:
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)

            # nseed = len(p1)
            self.p2_proxm = np.zeros((self.nsites))
            self.p2_secondary = np.zeros((self.nsites))
            for i in range(0, self.nsites):
                self.p2_dis = np.zeros((self.nseed))
                for j in range(0, self.nseed):
                    # self.p2_dis[j] = (math.sqrt((self.site_coordinates[i, 0] - self.corseed[j, 0]) ** 2 + (self.site_coordinates[i, 1] - self.corseed[j, 1]) ** 2))
                    self.p2_dis[j] = np.linalg.norm(self.site_coordinates[i] - self.corseed[j])
                self.p2_proxm[i] = np.argsort(self.p2_dis)[0] #get the membership in polony by taking the closest of distances to each seed
                self.p2_secondary[i] = np.argsort(self.p2_dis)[1] #get the second closest

            self.p2 = ["" for x in range(0, self.nsites)]
            for i in range(0, self.nsites):
                x = int(self.p2_proxm[i])
                self.p2[i] = reverse(self.p1[x])

            self.bc_index_pair = zip(self.p2_proxm, self.p2_secondary)

            # ipdb.set_trace()
            cmap = matplotlib.cm.get_cmap('Spectral')
            vor_polony = Voronoi(self.corseed, qhull_options = "")
            if full_output == True:
                if self.nseed <= 500:
                    voronoi_plot_2d(vor_polony,show_vertices=False)
                else:
                    voronoi_plot_2d(vor_polony, show_vertices=False,show_points = False, line_width = 0.1)
            self.rgb_stamp_catalog = np.zeros((self.nseed,3))
            if full_output == True:print 'begin gathering polony sizes'
            self.polony_sizes = collections.Counter(self.p2_proxm) #p2_proxm stores which seed each site is closest to
            # ipdb.set_trace()
            if full_output == True:print 'end of gathering polony sizes'
            for i in range(0, self.nseed):
                c = random.random()
                for j in range(0, self.nsites):
                    if self.p2_proxm[j] == i:
                        if full_output == True:
                            plt.scatter(self.site_coordinates[j, 0], self.site_coordinates[j, 1], color=cmap(c), s=1)
                        rgb, gray = get_pixel_rgb(self.scaled_imarray,self.lookup_x_axis,self.lookup_y_axis,self.site_coordinates[j,0],self.site_coordinates[j,1])
                        rgb = rgb / np.sum(rgb)
                        p_threshold = random.random()
                        if p_threshold < gray: #darker means closer to zero,
                            try:
                                colorID = np.random.choice([0,1,2],p=rgb)
                                self.rgb_stamp_catalog[i][colorID] += 1.
                            except:
                                self.rgb_stamp_catalog[i][0] += 1
                                self.rgb_stamp_catalog[i][1] += 1
                                self.rgb_stamp_catalog[i][2] += 1
                self.rgb_stamp_catalog[i] = self.rgb_stamp_catalog[i]/float(self.polony_sizes[i])
                # ipdb.set_trace()
            if full_output == True:print 'end of stamp cataloging'
            self.triangulated_delaunay = Delaunay(self.corseed, qhull_options="Qz Q2")

            self.ideal_graph = get_ideal_graph(self.triangulated_delaunay)

            ajc_polony = np.zeros((len(self.bc), len(self.bc)))
            ajc_polony.astype(int)
            for i in range(0, len(self.bc_index_pair)):
                rrr = int(self.bc_index_pair[i][0])
                ccc = int(self.bc_index_pair[i][1])
                ajc_polony[rrr, ccc] = 1
                ajc_polony[ccc, rrr] = 1

            if full_output==True: np.savetxt(self.directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")

            self.untethered_graph = nx.Graph()
            for i in range(0, ajc_polony.shape[0]):
                self.untethered_graph.add_node(i)
                for j in range(0, ajc_polony.shape[1]):
                    if ajc_polony[i][j] == 1:
                        self.untethered_graph.add_edge(i, j)


            if full_output == True:
                np.savetxt(self.directory_name + '/' + 'p2', self.p2, delimiter=",", fmt="%s")
                np.savetxt(self.directory_name + '/' + 'polony_seed', self.p2_proxm, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'point_adjacent_polony', self.p2_secondary, delimiter=",")
                np.savetxt(self.directory_name + '/' + 'bcindexpairlist', self.bc_index_pair, delimiter=',')
                # plt.rcParams['figure.figsize'] = (20,20)
                plt.xlim(-.1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.min(
                    self.site_coordinates[:, 0]),
                         .1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.max(
                             self.site_coordinates[:, 0]))
                plt.ylim(-.1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.min(
                    self.site_coordinates[:, 1]),
                         .1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.max(
                             self.site_coordinates[:, 1]))
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=0.5)

                # plt.figure(figsize=(10, 10), dpi=500)
                else:
                    plt.autoscale(enable=True, axis='both', tight=None)
                plt.savefig(self.directory_name+'/'+'polony.svg')
                plt.savefig(self.directory_name + '/' + 'polony.png')
                plt.close()
                plt.figure(figsize=(10, 10), dpi=500)
                plt.autoscale(enable=True, axis='both', tight=None)
                plt.triplot(self.corseed[:,0],self.corseed[:,1],self.triangulated_delaunay.simplices.copy(),linestyle='solid',alpha=1,marker=None,color='#3498DB')
                if self.nseed <= 500:
                    plt.plot(self.corseed[:,0],self.corseed[:,1],'o',color='#3498DB',markerfacecolor='w',markersize=12)
                # plt.rcParams['figure.figsize'] = (20, 20)
                if self.nseed <= 500:
                    for i in range(0, self.nseed):
                        plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=1, fontsize=6,horizontalalignment='center', verticalalignment='center')
                plt.savefig(self.directory_name + '/' +'delaunay_polony.svg')
                plt.savefig(self.directory_name + '/' + 'delaunay_polony.png')
                plt.close()
                plt.autoscale(enable=False, axis='both', tight=None)
            print 'end of crosslink polonies'




        print 'end of crosslink polonies'

            # return p2, p2_secondary, p2_proxm, bc_index_pair, tri, rgb_stamp_catalog

    def check_continuity(self):
        if (planarity.is_planar(self.untethered_graph)) == True and nx.is_connected(self.untethered_graph) == True:
            self.disconnected = False
        else:
            self.disconnected = True

    # def crosslink_poloniesbackup(self, full_output=True):
    #     if full_output == True:
    #         plt.close()
    #         plt.figure(figsize=(10, 10), dpi=500)
    #
    #     # nseed = len(p1)
    #     self.p2_proxm = np.zeros((self.nsites))
    #     self.p2_secondary = np.zeros((self.nsites))
    #     for i in range(0, self.nsites):
    #         self.p2_dis = np.zeros((self.nseed))
    #         for j in range(0, self.nseed):
    #             # self.p2_dis[j] = (math.sqrt((self.site_coordinates[i, 0] - self.corseed[j, 0]) ** 2 + (self.site_coordinates[i, 1] - self.corseed[j, 1]) ** 2))
    #             self.p2_dis[j] = np.linalg.norm(self.site_coordinates[i] - self.corseed[j])
    #         self.p2_proxm[i] = np.argsort(self.p2_dis)[0] #get the membership in polony by taking the closest of distances to each seed
    #         self.p2_secondary[i] = np.argsort(self.p2_dis)[1] #get the second closest
    #
    #     self.p2 = ["" for x in range(0, self.nsites)]
    #     for i in range(0, self.nsites):
    #         x = int(self.p2_proxm[i])
    #         self.p2[i] = reverse(self.p1[x])
    #
    #     self.bc_index_pair = zip(self.p2_proxm, self.p2_secondary)
    #
    #     # ipdb.set_trace()
    #     cmap = matplotlib.cm.get_cmap('Spectral')
    #     vor_polony = Voronoi(self.corseed, qhull_options = "")
    #     if full_output == True:
    #         if self.nseed <= 500:
    #             voronoi_plot_2d(vor_polony,show_vertices=False)
    #         else:
    #             voronoi_plot_2d(vor_polony, show_vertices=False,show_points = False, line_width = 0.1)
    #     self.rgb_stamp_catalog = np.zeros((self.nseed,3))
    #     if full_output == True:print 'begin gathering polony sizes'
    #     self.polony_sizes = collections.Counter(self.p2_proxm) #p2_proxm stores which seed each site is closest to
    #     ipdb.set_trace()
    #     if full_output == True:print 'end of gathering polony sizes'
    #     for i in range(0, self.nseed):
    #         c = random.random()
    #         for j in range(0, self.nsites):
    #             if self.p2_proxm[j] == i:
    #                 if full_output == True:
    #                     plt.scatter(self.site_coordinates[j, 0], self.site_coordinates[j, 1], color=cmap(c), s=1)
    #                 rgb, gray = get_pixel_rgb(self.scaled_imarray,self.lookup_x_axis,self.lookup_y_axis,self.site_coordinates[j,0],self.site_coordinates[j,1])
    #                 rgb = rgb / np.sum(rgb)
    #                 p_threshold = random.random()
    #                 if p_threshold < gray: #darker means closer to zero,
    #                     try:
    #                         colorID = np.random.choice([0,1,2],p=rgb)
    #                         self.rgb_stamp_catalog[i][colorID] += 1.
    #                     except:
    #                         self.rgb_stamp_catalog[i][0] += 1
    #                         self.rgb_stamp_catalog[i][1] += 1
    #                         self.rgb_stamp_catalog[i][2] += 1
    #         self.rgb_stamp_catalog[i] = self.rgb_stamp_catalog[i]/float(self.polony_sizes[i])
    #         # ipdb.set_trace()
    #     if full_output == True:print 'end of stamp cataloging'
    #     self.triangulated_delaunay = Delaunay(self.corseed, qhull_options="Qz Q2")
    #
    #     self.ideal_graph = get_ideal_graph(self.triangulated_delaunay)
    #
    #     ajc_polony = np.zeros((len(self.bc), len(self.bc)))
    #     ajc_polony.astype(int)
    #     for i in range(0, len(self.bc_index_pair)):
    #         rrr = int(self.bc_index_pair[i][0])
    #         ccc = int(self.bc_index_pair[i][1])
    #         ajc_polony[rrr, ccc] = 1
    #         ajc_polony[ccc, rrr] = 1
    #
    #     if full_output==True: np.savetxt(self.directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")
    #
    #     self.untethered_graph = nx.Graph()
    #     for i in range(0, ajc_polony.shape[0]):
    #         self.untethered_graph.add_node(i)
    #         for j in range(0, ajc_polony.shape[1]):
    #             if ajc_polony[i][j] == 1:
    #                 self.untethered_graph.add_edge(i, j)
    #
    #
    #     if full_output == True:
    #         np.savetxt(self.directory_name + '/' + 'p2', self.p2, delimiter=",", fmt="%s")
    #         np.savetxt(self.directory_name + '/' + 'polony_seed', self.p2_proxm, delimiter=",")
    #         np.savetxt(self.directory_name + '/' + 'point_adjacent_polony', self.p2_secondary, delimiter=",")
    #         np.savetxt(self.directory_name + '/' + 'bcindexpairlist', self.bc_index_pair, delimiter=',')
    #         # plt.rcParams['figure.figsize'] = (20,20)
    #         plt.xlim(-.1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.min(
    #             self.site_coordinates[:, 0]),
    #                  .1 * (np.max(self.site_coordinates[:, 0]) - np.min(self.site_coordinates[:, 0])) + np.max(
    #                      self.site_coordinates[:, 0]))
    #         plt.ylim(-.1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.min(
    #             self.site_coordinates[:, 1]),
    #                  .1 * (np.max(self.site_coordinates[:, 1]) - np.min(self.site_coordinates[:, 1])) + np.max(
    #                      self.site_coordinates[:, 1]))
    #         if self.nseed <= 500:
    #             for i in range(0, self.nseed):
    #                 plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=0.5)
    #
    #         # plt.figure(figsize=(10, 10), dpi=500)
    #         else:
    #             plt.autoscale(enable=True, axis='both', tight=None)
    #         plt.savefig(self.directory_name+'/'+'polony.svg')
    #         plt.savefig(self.directory_name + '/' + 'polony.png')
    #         plt.close()
    #         plt.figure(figsize=(10, 10), dpi=500)
    #         plt.autoscale(enable=True, axis='both', tight=None)
    #         plt.triplot(self.corseed[:,0],self.corseed[:,1],self.triangulated_delaunay.simplices.copy(),linestyle='solid',alpha=1,marker=None,color='#3498DB')
    #         if self.nseed <= 500:
    #             plt.plot(self.corseed[:,0],self.corseed[:,1],'o',color='#3498DB',markerfacecolor='w',markersize=12)
    #         # plt.rcParams['figure.figsize'] = (20, 20)
    #         if self.nseed <= 500:
    #             for i in range(0, self.nseed):
    #                 plt.text(self.corseed[i, 0], self.corseed[i, 1], '%s' % i, alpha=1, fontsize=6,horizontalalignment='center', verticalalignment='center')
    #         plt.savefig(self.directory_name + '/' +'delaunay_polony.svg')
    #         plt.savefig(self.directory_name + '/' + 'delaunay_polony.png')
    #         plt.close()
    #         plt.autoscale(enable=False, axis='both', tight=None)
    #     print 'end of crosslink polonies'




class Reconstruction:
    def __init__(self, directory_name, corseed, ideal_graph, untethered_graph, rgb_stamp_catalog):
        self.directory_name = directory_name
        self.corseed = corseed
        self.nseed = len(corseed)
        self.ideal_graph = ideal_graph
        self.untethered_graph = untethered_graph
        self.rgb_stamp_catalog = rgb_stamp_catalog

    def conduct_pca_peripheral(self,full_output=False,image_only=False):
        plt.close()
        self.npolony = len(self.untethered_graph.nodes)
        face_enumeration_t0 = time.time()
        faces = self.fast_planar_embedding(self.untethered_graph, title='tutte', full_output=False)



        self.face_enumeration_runtime = time.time() - face_enumeration_t0
        self.max_face = []
        for face in range(0, len(faces)):
            if len(faces[face]) > len(self.max_face):
                self.max_face = faces[face]
        peripheral_nodes_list = np.fromiter(itertools.chain.from_iterable(self.max_face), dtype=int)

        self.peripheral_nodes = set(peripheral_nodes_list)
        # length_of_outer_face = len(self.peripheral_nodes)
        # tridistance = int(length_of_outer_face/10)
        # self.equidistant_three_nodes = [peripheral_nodes_list[0], peripheral_nodes_list[tridistance], peripheral_nodes_list[2*tridistance]]
        self.distance_matrix = np.zeros((self.npolony,len(self.peripheral_nodes)), dtype=int)
        # csr_graph = csr_matrix(nx.adjacency_matrix(self.untethered_graph))
        # self.distance_matrix = np.array(floyd_warshall(csgraph=csr_graph, directed=False, return_predecessors=True))[0]
        #
        for node in self.untethered_graph.nodes:
            peripheral_node_count = 0
            for peripheral_node in self.peripheral_nodes:
                # print node, peripheral_node
                # distance_matrix[node,peripheral_node] = \
                shortest_paths = [p for p in nx.algorithms.shortest_paths.generic.all_shortest_paths(self.untethered_graph,source=node,target=peripheral_node)]
                self.distance_matrix[node,peripheral_node_count] = len(shortest_paths[0])
                peripheral_node_count += 1
        # ipdb.set_trace()
        pca = PCA(n_components = 2)
        principal_components = pca.fit_transform(self.distance_matrix)
        self.reconstructed_pos = dict(zip(self.untethered_graph.nodes, tuple(principal_components)))
        self.reconstructed_points = np.array(convert_dict_to_list(self.reconstructed_pos))
        self.reconstructed_points = self.reconstructed_points / np.max(np.sqrt(self.reconstructed_points ** 2))
        # ipdb.set_trace()
        # self.reconstructed_pos = 1
        # self.reconstructed_points = np.array(convert_dict_to_list(self.reconstructed_pos))
        # self.rgb_stamp_catalog = (self.rgb_stamp_catalog)
        # # maxima = np.max(self.rgb_stamp_catalog)
        # # self.rgb_stamp_catalog = self.rgb_stamp_catalog / maxima
        if full_output == True or image_only == True:

            draw_embedding_diagram(self.untethered_graph, self.reconstructed_points, self.directory_name, nseed=self.nseed, graph_name='embedding_graph_topodist_peripheral')

            draw_voronoi_image(self.reconstructed_points, self.rgb_stamp_catalog, self.directory_name, nseed = self.nseed, voronoi_name='voronoi_topodist_peripheral')



    def conduct_shortest_path_matrix(self,full_output=False,image_only=False):
        plt.close()
        self.npolony = len(self.untethered_graph.nodes)
        csr_graph = csr_matrix(nx.adjacency_matrix(self.untethered_graph))
        self.distance_matrix = np.array(floyd_warshall(csgraph=csr_graph, directed=False, return_predecessors=True))[0]
        maxdistance = np.max(self.distance_matrix)
        # self.distance_matrix[self.distance_matrix < 0.45*maxdistance] = 0
        pca = PCA(n_components = 2)
        principal_components = pca.fit_transform(self.distance_matrix)
        self.reconstructed_pos = dict(zip(self.untethered_graph.nodes, tuple(principal_components)))
        self.reconstructed_points = np.array(principal_components)
        self. reconstructed_points = self.reconstructed_points / np.max(np.sqrt(self.reconstructed_points ** 2))



        # self.rgb_stamp_catalog = (self.rgb_stamp_catalog)
        # maxima = np.max(self.rgb_stamp_catalog)
        # self.rgb_stamp_catalog = self.rgb_stamp_catalog / maxima
        if full_output == True or image_only == True:

            draw_embedding_diagram(self.untethered_graph, self.reconstructed_points, self.directory_name, nseed=self.nseed,
                                   graph_name='embedding_graph_topodist_total')
            draw_voronoi_image(self.reconstructed_points, self.rgb_stamp_catalog, self.directory_name,nseed = self.nseed,
                               voronoi_name='voronoi_topodist_total')



    def conduct_tutte_embedding_from_ideal(self,full_output=False,image_only=False):

        plt.close()

        #check for planarity
        # metrics are removed however
        if (planarity.is_planar(self.ideal_graph))==True and nx.is_connected(self.ideal_graph)==True:
            if full_output == True:
                ideal_faces = self.fast_planar_embedding(self.ideal_graph,title='ideal')
                # plt.autoscale(enable=True, axis='both', tight=None)
                # plt.savefig(self.directory_name + '/' + 'ideal_planar.png')
            else:
                ideal_faces = self.fast_planar_embedding(self.ideal_graph, title='ideal',full_output=False)
        else:
            print 'error, graph not fully connected, planarity test failed'
            ipdb.set_trace()
        self.ideal_max_face = []
        for face in range(0, len(ideal_faces)):
            if len(ideal_faces[face]) > len(self.ideal_max_face):
                self.ideal_max_face = ideal_faces[face]
        self.ideal_pos = tutte_embedding(self.ideal_graph, self.ideal_max_face)
        self.reconstructed_points = np.array(convert_dict_to_list(self.ideal_pos))
        if full_output == True:

            # draw_embedding_diagram(self.untethered_graph, self.reconstructed_points, self.directory_name,nseed=self.nseed,
            #                        graph_name='embedding_graph_tutte_ideal')
            draw_voronoi_image(self.reconstructed_points, self.rgb_stamp_catalog, self.directory_name,nseed = self.nseed,
                               voronoi_name='voronoitutteideal')

    def conduct_spring_embedding(self,full_output=True,image_only=False):

        plt.close()
        self.reconstructed_pos = nx.drawing.nx_agraph.pygraphviz_layout(self.untethered_graph, args='-Gepsilon=0.0000000001')
        #nx.draw_networkx_edges(self.untethered_graph, self.reconstructed_pos, alpha=0.3)
        #ipdb.set_trace()
        self.reconstructed_points = np.array(convert_dict_to_list(self.reconstructed_pos))
        normalizing_factor = np.max(np.sqrt(self.reconstructed_points ** 2))
        self.reconstructed_points = self.reconstructed_points / normalizing_factor
        # self.rgb_stamp_catalog = (self.rgb_stamp_catalog)
        # maxima = np.max(self.rgb_stamp_catalog)
        # self.rgb_stamp_catalog = self.rgb_stamp_catalog / maxima
        # self.untethered_graph.add_edges_from(triangulation_fix(faces, self.reconstructed_pos))
        if full_output == True or image_only == True:

            draw_embedding_diagram(self.untethered_graph, self.reconstructed_points, self.directory_name,nseed=self.nseed,
                                   graph_name='embedding_graph_spring')

            draw_voronoi_image(self.reconstructed_points, self.rgb_stamp_catalog, self.directory_name,nseed = self.nseed,
                               voronoi_name='voronoi_spring')



    def conduct_tutte_embedding(self,full_output=True,image_only=False):

        plt.close()

        # blind reconstruction here, if there are missed edges, then they will distort final reconstruction
        face_enumeration_t0 = time.time()
        # if (planarity.is_planar(self.untethered_graph)) == True and nx.is_connected(self.untethered_graph) == True:
        #     if full_output == True:
        #         faces = self.fast_planar_embedding(self.untethered_graph, title='tutte')
        #     else:
        faces = self.fast_planar_embedding(self.untethered_graph, title='tutte', full_output=False)
        # else:
        #     print 'error, graph not fully connected, planarity test failed'
        #     ipdb.set_trace()
        self.face_enumeration_runtime = time.time() - face_enumeration_t0
        self.max_face = []

        for face in range(0, len(faces)):
            if len(faces[face]) > len(self.max_face):
                self.max_face = faces[face]
        self.reconstructed_pos = tutte_embedding(self.untethered_graph, self.max_face)  # need to update self.untethered_graph
        self.reconstructed_points = np.array(convert_dict_to_list(self.reconstructed_pos))
        ###########
        # if len(self.untethered_graph) != len(self.reconstructed_points):
        #     ipdb.set_trace()
        # self.untethered_graph.add_edges_from(triangulation_fix(faces,reconstructed_pos))


        # self.updated_reconstructed_pos = self.tutte_embedding(self.untethered_graph, self.max_face)
        if full_output == True or image_only == True:
            draw_embedding_diagram(self.untethered_graph, self.reconstructed_points, self.directory_name,nseed=self.nseed,
                                   graph_name='embedding_graph_tutte')

            draw_voronoi_image(self.reconstructed_points, self.rgb_stamp_catalog, self.directory_name,nseed = self.nseed,
                               voronoi_name='voronoi_tutte')




    def fast_planar_embedding(self, graph, labels=True, title='',full_output=True):
        pgraph = planarity.PGraph(graph)
        pgraph.embed_drawplanar()
        hgraph = planarity.networkx_graph(pgraph) #returns a networkx graph built from planar graph- nodes and edges only
        patches = []
        node_labels = {}
        xs = []
        ys = []
        all_vlines = []
        xmax=0
        ordered_neighbor_lists = [None]*graph.number_of_nodes()

        for node in range(0,graph.number_of_nodes()):
            try:
                # ordered_clockwise_neighbors = []
                upper_neighbors = []
                lower_neighbors = []
                upper_neighbor_positions = []
                lower_neighbor_positions = []
                neighbors = graph[node]
                position_of_this_node = hgraph.nodes(data=False)[node]['pos']
                for neighbor in neighbors:
                    position_of_this_neighbor = hgraph.nodes(data=False)[neighbor]['pos']
                    if position_of_this_neighbor > position_of_this_node:
                        upper_neighbors.append(neighbor)
                        upper_neighbor_positions.append(hgraph.edges(data=False)[(node, neighbor)]['pos'])
                    else:
                        lower_neighbors.append(neighbor)
                        lower_neighbor_positions.append(hgraph.edges(data=False)[(node, neighbor)]['pos'])
                upper_neighbors_sorted = [x for _,x in sorted(zip(upper_neighbor_positions, upper_neighbors))]
                lower_neighbors_sorted = [x for _, x in sorted(zip(lower_neighbor_positions, lower_neighbors))]
                ordered_neighbor_lists[node] = upper_neighbors_sorted[::-1] + lower_neighbors_sorted
            except:
                # ipdb.set_trace()
                print 'error during face enumeration, node not found in graph'
                pass
        edge_list_ordered_pairs = []
        for edge in graph.edges():
            edge_list_ordered_pairs.append([edge[0], edge[1]])
            edge_list_ordered_pairs.append([edge[1], edge[0]])
        face_list = []
        while edge_list_ordered_pairs:
            arbitrary_edge = edge_list_ordered_pairs[0]
            # ipdb.set_trace()
            vv = arbitrary_edge[1]
            uu = arbitrary_edge[0]
            u_0 = arbitrary_edge[0]
            face = [uu]
            edge_list_ordered_pairs.remove(arbitrary_edge)
            while vv != u_0:
                face.append(vv)
                nbrs_v = ordered_neighbor_lists[vv]
                # ipdb.set_trace()
                index_of_uu = nbrs_v.index(uu)
                index_of_ww = index_of_uu + 1 #ww is the clockwise successor of uu
                ww = nbrs_v[index_of_ww%len(nbrs_v)]
                try:edge_list_ordered_pairs.remove([vv,ww])
                except:
                    print 'error during face enumeration, edge not in list of ordered pairs'
                    pass
                uu = vv
                vv = ww
            face_list.append(face)
        face_edge_list = []
        for face in face_list:
            converted_face = []
            for vertex in range(len(face)):
                successor = (vertex+1)%len(face)
                edge_to_add = (face[vertex],face[successor])
                converted_face.append(edge_to_add)
            face_edge_list.append(converted_face)

        return face_edge_list





    def align(self,title='alignment',error_threshold = 50,edge_number = 0, full_output = True,image_only=False,get_average_distortogram=False):
        plt.close()
        minerror = 99999999
        if edge_number == 0:
            edge_number = len(self.corseed)
        minevals = 0
        while minerror > error_threshold and minevals < 1000:
            self.corseed = self.corseed * np.random.choice([-1.,1.],2) #generates a 2 item list of either -1s or +1s to flop coordinates
            initialize = np.array([random.uniform(0.,6.3),random.uniform(-10, 10.),random.uniform(-10, 10.),random.uniform(0.,1)])
            res1 = scipy.optimize.basinhopping(func=evaluate_distortion,
                                              x0=initialize, T=5.0,
                                              minimizer_kwargs={'args':([self.corseed, self.reconstructed_points, False,edge_number],)},
                                              niter=1,
                                              disp=True,
                                              stepsize = 1,
                                              niter_success=1,
                                               accept_test = None,
                                              callback=callback_on_optimization_indicator)

            if minerror > res1['fun']:
                minerror = res1['fun']
            else:
                pass
            minevals += 1

        xopt1 = res1['x']
        error1, self.corseed_adjusted = evaluate_distortion(xopt1,args=[self.corseed, self.reconstructed_points,True,True,edge_number])

        if full_output == True:
            plt.close()
            plt.figure(figsize=(10, 10), dpi=500)
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)
            plt.scatter(self.corseed_adjusted[:,0],self.corseed_adjusted[:,1])
            plt.scatter(self.reconstructed_points[:,0],self.reconstructed_points[:,1])
        self.distances = []
        for i in range(0, len(self.reconstructed_points)):
            if full_output == True:
                plt.text(self.corseed_adjusted[i, 0], self.corseed_adjusted[i, 1], '%s' % i, alpha=0.5)
                plt.text(self.reconstructed_points[i, 0], self.reconstructed_points[i, 1], '%s' % i, alpha=0.5)
                plt.plot([self.corseed_adjusted[i, 0],self.reconstructed_points[i, 0]],[self.corseed_adjusted[i, 1],self.reconstructed_points[i, 1]],'k-')
            # self.distances += [np.sqrt((self.corseed_adjusted[i, 0]-self.reconstructed_points[i, 0])**2+(self.corseed_adjusted[i, 1]-self.reconstructed_points[i, 1])**2)]
            self.distances += [np.linalg.norm(self.corseed_adjusted[i] - self.reconstructed_points[i])]

        #rescale for average distortogram
        if get_average_distortogram == True:
            discretized_distortogram_dimension = 25
            # distances_from_origin = np.zeros((len(self.corseed_adjusted)),dtype=np.float)
            # for i in range(0,len(distances_from_origin)):
            distances_from_origin = [np.linalg.norm(i) for i in self.corseed_adjusted]


            self.corseed_for_avgdistortogram = np.zeros((len(self.corseed_adjusted),2))
            self.corseed_for_avgdistortogram = self.corseed_adjusted/np.max(distances_from_origin)
            # xcoreseedrange = np.max(self.corseed_for_avgdistortogram[:,0]) - np.min(self.corseed_for_avgdistortogram[:,0])
            # ycoreseedrange = np.max(self.corseed_for_avgdistortogram[:, 1]) - np.min(self.corseed_for_avgdistortogram[:, 1])
            self.corseed_for_avgdistortogram = (self.corseed_for_avgdistortogram + 1)/2.
            # self.corseed_for_avgdistortogram[:,0] =( self.corseed_for_avgdistortogram[:,0] + 1.)/2.
            # self.corseed_for_avgdistortogram[:, 1] = (self.corseed_for_avgdistortogram[:, 1] + 1.) / 2.
            self.discretized_distortogram = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
            self.discretized_distortogram_counter = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
            reconstructed_points_discretized = np.array(np.ceil(self.corseed_for_avgdistortogram*discretized_distortogram_dimension)-1, dtype=int)
            # ipdb.set_trace()
            for i in range(0, len(reconstructed_points_discretized)):
                self.discretized_distortogram_counter[reconstructed_points_discretized[i][0], reconstructed_points_discretized[i][1]] += 1.0
                try: self.discretized_distortogram[reconstructed_points_discretized[i][0], reconstructed_points_discretized[i][1]] += self.distances[i]
                except: pass
        # ipdb.set_trace()
        if full_output == True:

            tri1 = Delaunay(self.corseed_adjusted)
            plt.triplot(self.corseed_adjusted[:, 0], self.corseed_adjusted[:, 1], tri1.simplices.copy(),'b-',alpha=0.2,marker='')
            if self.untethered_graph == 'None':
                tri2 = Delaunay(self.reconstructed_points)
                plt.triplot(self.reconstructed_points[:, 0], self.reconstructed_points[:, 1], tri2.simplices.copy(),'r-',alpha=0.2,marker='')
            else:
                nx.draw_networkx_edges(self.untethered_graph,self.reconstructed_points,width=1.0,alpha=0.2,edge_color='r')
                nx.draw_networkx_nodes(self.untethered_graph, self.reconstructed_points,node_color='r',node_size=25,alpha=0.2)

            plt.savefig(self.directory_name+'/'+ title +'alignment_diagram.svg')
            plt.savefig(self.directory_name + '/' + title + 'alignment_diagram.png')
            plt.close()

        if full_output == True or image_only == True:
            plt.close()
            plt.figure(figsize=(10,10), dpi=500)
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1, 1.1)
            cmap = matplotlib.cm.get_cmap('magma_r')
            # normalize = matplotlib.colors.Normalize(vmin=min(self.distances), vmax=max(self.distances))
            normalize = matplotlib.colors.Normalize(vmin = 0, vmax = 2.0)
            colors = [cmap(normalize(value)) for value in self.distances]

            if len(self.corseed_adjusted) < 200:
                plt.scatter(self.corseed_adjusted[:,0],self.corseed_adjusted[:,1],c='k', alpha=0.1)
                plt.scatter(self.reconstructed_points[:,0],self.reconstructed_points[:,1],c='k',marker='o', alpha=0.1)
            for i in range(0, len(self.reconstructed_points)):
                plt.plot([self.corseed_adjusted[i, 0],self.reconstructed_points[i, 0]],[self.corseed_adjusted[i, 1],self.reconstructed_points[i, 1]],'-',color=colors[i],linewidth=10.0)
            # ipdb.set_trace()
            # plt.colorbar()
            plt.savefig(self.directory_name+'/'+ title +'distortogram.svg')
            plt.savefig(self.directory_name + '/' + title + 'distortogram.png')
            plt.close()
            print 'final cost function value: ',minerror

            fig, ax = plt.subplots(figsize = (6,1))
            fig.subplots_adjust(bottom = 0.5)
            cb1 = matplotlib.colorbar.ColorbarBase(ax,cmap=cmap,norm=normalize,orientation='horizontal')
            plt.savefig(self.directory_name + '/' + title + 'distortogram_colorbar.svg')
            plt.savefig(self.directory_name + '/' + title + 'distortogram_colorbar.png')
            plt.close()

























    def get_delaunay_comparison(self,flat_pos_list,untethered_graph,name='',full_output = False):
        # ipdb.set_trace()
        pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
        # b = dict(zip(a[::2], a[1::2]))
        # ipdb.set_trace()
        pos = dict(zip(np.arange(len(pos_list)), pos_list))
        new_delaunay = Delaunay(pos_list)
        new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
        old_topology =[e for e in untethered_graph.edges()]
        self.good_edges = [edge for edge in new_topology if edge in old_topology]
        self.bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
        self.missed_old_edges = [edge for edge in old_topology if edge not in new_topology]
        pos_x, pos_y = zip(*pos_list)[0],zip(*pos_list)[1]
        number_of_bad_edges = len(self.bad_new_edges)
        number_of_missed_edges = len(self.missed_old_edges)
        total_badedges = number_of_bad_edges + number_of_missed_edges
        # average_distance = 0
        # normalizing_factor = len(new_topology)
        # missing_edge_penalty = 1
        # bad_edge_penalty = len(bad_new_edges)
        # good_edge_reward = 1
        # for edge in new_topology:
        #     average_distance += get_euclidean_edgelength(edge,pos)
        # average_distance = average_distance / len(new_topology)
        # for edge in missed_old_edges:
        #     missing_edge_penalty += get_euclidean_edgelength(edge,pos)
        # for edge in good_edges:
        #     good_edge_reward += get_euclidean_edgelength(edge, pos)
        # for edge in bad_new_edges:
        #     bad_edge_penalty += get_euclidean_edgelength(edge, pos)

        if full_output == True:
            plt.triplot(pos_x,pos_y,new_delaunay.simplices.copy(),linestyle=':')
            nx.draw_networkx(untethered_graph,pos=pos_list,linestyle=':',width=1.0,alpha=0.2,edge_color='r',node_color='r',node_size=30)
            plt.plot(pos_x,pos_y,'o')
            plt.rcParams['figure.figsize'] = (20, 20)
            for i in range(0, len(pos_list)):
                plt.text(pos_x[i], pos_y[i], '%s' % i, alpha=0.5)
            plt.xlim(-1.1,1.1)
            plt.ylim(-1.1, 1.1)
            plt.savefig(self.directory_name + '/' +'delaunay_badedges_'+str(total_badedges)+str(name)+'.svg')
            plt.savefig(self.directory_name + '/' + 'delaunay_badedges_'+str(total_badedges)+str(name)+'.png')
            plt.close()

        cost = (len(self.bad_new_edges) + len(self.missed_old_edges)) / len(new_topology)
        self.n_bad_new_edges = len(self.bad_new_edges)
        self.n_missed_old_edges = len(self.missed_old_edges)
        self.n_good_edges = len(self.good_edges)
        self.average_isotopy = (self.n_good_edges*(len(new_topology)+len(old_topology)))/(2*len(new_topology)*len(old_topology))
        self.levenshtein_distance = self.n_bad_new_edges + self.n_missed_old_edges
        self.normd_levenshtein_distance = float(self.levenshtein_distance) / float(len(new_topology))
        # return good_edges,bad_new_edges,missed_old_edges,cost

    def get_delaunay_error(self,flat_pos_list,untethered_graph):
        # ipdb.set_trace()
        pos_list = zip(flat_pos_list[::2], flat_pos_list[1::2])
        # b = dict(zip(a[::2], a[1::2]))
        pos = dict(zip(np.arange(len(pos_list)), pos_list))
        new_delaunay = Delaunay(pos_list)
        new_topology=[e for e in get_ideal_graph(new_delaunay).edges()]
        old_topology =[e for e in untethered_graph.edges()]
        good_edges = [edge for edge in new_topology if edge in old_topology]
        bad_new_edges = [edge for edge in new_topology if edge not in old_topology]
        missed_old_edges = [edge for edge in old_topology if edge not in new_topology]
        # ipdb.set_trace()
        average_distance = 0
        missing_edge_penalty = 1
        bad_edge_penalty = 1
        # good_edge_reward = 1
        for edge in new_topology:
            average_distance += get_euclidean_edgelength(edge,pos)
        average_distance = average_distance/len(new_topology)
        for edge in missed_old_edges:
            missing_edge_penalty += get_euclidean_edgelength(edge,pos)/len(missed_old_edges)
        # for edge in good_edges:
        #     good_edge_reward += get_euclidean_edgelength(edge, pos)
        for edge in bad_new_edges:
            bad_edge_penalty += get_euclidean_edgelength(edge, pos)/len(bad_new_edges)

        cost = (len(bad_new_edges)*bad_edge_penalty+len(missed_old_edges)*missing_edge_penalty)/(len(new_topology)*average_distance)
        return cost

    def anneal_to_initial_delaunay(self,pos, untethered_graph, T=0.1,niter=1,name=''):
        pos_list= positiondict_to_list(pos)[0]
        # error = get_delaunay_error(flat_pos_list=tutte_pos_list,untethered_graph=UntetheredG)
        minimizer_kwargs = {"tol":10000, "args": (untethered_graph,)}
        res = scipy.optimize.basinhopping(func=get_delaunay_error,
                                          x0=pos_list, T=T,
                                          minimizer_kwargs=minimizer_kwargs,
                                          niter=niter,
                                          disp = True,
                                          niter_success=50000,
                                          callback=callback_on_optimization_indicator)
        annealed_pos_list = zip(res.x[::2], res.x[1::2])
        annealed_pos = dict(zip(np.arange(len(annealed_pos_list)), annealed_pos_list))
        vor_reconstructed = Voronoi(convert_dict_to_list(annealed_pos))
        voronoi_plot_2d(vor_reconstructed, show_vertices=False,show_points=False)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.savefig(self.directory_name + '/' + 'annealed_voronoi'+str(name)+'.svg')
        plt.savefig(self.directory_name + '/' + 'annealed_voronoi'+str(name)+'.png')
        plt.close()

        good_edges, bad_edges, missed_edges, cost = get_delaunay_comparison(pos_list, untethered_graph, name=name + 'before')
        good_edges_annealed, bad_edges_annealed, missed_edges_annealed, cost_annealed = get_delaunay_comparison(res.x, untethered_graph, name=name+'annealed')
        print 'input position discrepancies:'
        print 'missed edges: ' + str(missed_edges)
        print 'false edges: ' + str(bad_edges)
        print 'annealed position discrepancies:'
        print 'missed edges: ' + str(missed_edges_annealed)
        print 'false edges: ' + str(bad_edges_annealed)
        return annealed_pos










































































##########################
###### Subroutines #######
##########################
def draw_embedding_diagram(untethered_graph, reconstructed_points, directory_name,nseed, graph_name = 'embedding_diagram'):
    plt.close()
    plt.autoscale(enable=False, axis='both', tight=None)
    plt.figure(figsize=(10, 10), dpi=500)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks(np.arange(-1, 1.25, 0.25))
    plt.yticks(np.arange(-1, 1.25, 0.25))
    try:nx.draw_networkx_edges(untethered_graph, reconstructed_points, alpha=1.0,edge_color='#E74C3C')
    except:ipdb.set_trace()
    if nseed <= 500:
        nodes = nx.draw_networkx_nodes(untethered_graph, reconstructed_points, alpha=1.0, node_color='w',linewidths=0.3, node_size = 150)
        nodes.set_edgecolor('#E74C3C')
        nx.draw_networkx_labels(untethered_graph, reconstructed_points, font_family='Times', font_size=6, alpha=1)
    plt.savefig(directory_name + '/' + graph_name + '.svg')
    plt.savefig(directory_name + '/' + graph_name + '.png')
    plt.close()

def draw_voronoi_image(reconstructed_points, rgb_stamp_catalog, directory_name, nseed, voronoi_name = 'voronoi'):
    rgb_stamp_catalog = (rgb_stamp_catalog)
    collapse = np.sum(rgb_stamp_catalog, axis=1)
    maxima = np.max(collapse)
    rgb_stamp_catalog = rgb_stamp_catalog / (maxima/3.0)
    rgb_stamp_catalog[rgb_stamp_catalog > 1.0 ] = 1.0
    plt.close()
    # plt.autoscale(enable=False, axis='both', tight=None)
    plt.figure(figsize=(10, 10), dpi=500)
    # plt.xlim(-1.1, 1.1)
    # plt.ylim(-1.1, 1.1)
    # plt.xticks(np.arange(-1, 1, 0.25))
    # plt.yticks(np.arange(-1, 1, 0.25))
    # ipdb.set_trace()
    vor_reconstructed_spring = Voronoi(reconstructed_points)
    if nseed <= 500:
        voronoi_plot_2d(vor_reconstructed_spring, show_vertices=False, show_points=False, line_alpha=0.1)
    else:
        voronoi_plot_2d(vor_reconstructed_spring, show_vertices=False, show_points=False, line_alpha=0.0)
    for r in range(len(vor_reconstructed_spring.point_region)):
        region = vor_reconstructed_spring.regions[vor_reconstructed_spring.point_region[r]]
        if not -1 in region:
            polygon = [vor_reconstructed_spring.vertices[i] for i in region]
            # ipdb.set_trace()
            plt.fill(*zip(*polygon),edgecolor=None,linewidth=0,linestyle='--',
                     color=(rgb_stamp_catalog[r][0], rgb_stamp_catalog[r][1], rgb_stamp_catalog[r][2]))
    # ipdb.set_trace()
    # ipdb.set_trace()
    plt.autoscale(enable=False, axis='both', tight=None)
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.xticks(np.arange(-1, 1.25, 0.25))
    plt.yticks(np.arange(-1, 1.25, 0.25))
    plt.rcParams['figure.figsize'] = (5, 5)
    plt.savefig(directory_name + '/' + voronoi_name + '.svg')
    plt.savefig(directory_name + '/' + voronoi_name + '.png')
    plt.close()

def prefix(directory_label):
    # constructs a directory name that will be propagated to all file saving events
    global directory_name
    global today
    today = datetime.date.today()
    directory_name = str(today) + '_'+ str(directory_label)+ '_' + str(time.time())
    os.makedirs(directory_name)
    return directory_name

def tutte_embedding(graph, outer_face):
    pos = {}  # a dictionary of node positions
    tmp = nx.Graph()
    for edge in outer_face:
        aaa, bbb = edge
        tmp.add_edge(aaa, bbb)

    tmp_pos = circular_layout_sorted(tmp,1.0)  # ensures that outterface is a convex shape
    pos.update(tmp_pos)
    outer_vertices = tmp.nodes()
    remaining_vertices = [x for x in graph.nodes() if x not in outer_vertices]
    size = len(remaining_vertices)
    AAA = [[0 for i in range(size)] for i in range(
        size)]  # create the the system of equations that will determine the x and y positions of remaining vertices
    bbb = [0 for i in range(size)]  # the elements of theses matrices are indexed by the remaining_vertices list
    CCC = [[0 for i in range(size)] for i in range(size)]
    ddd = [0 for i in range(size)]
    for u in remaining_vertices:
        i = remaining_vertices.index(u)
        neighbors = [node for node in graph.neighbors(u)]
        n = len(neighbors)
        AAA[i][i] = 1
        CCC[i][i] = 1
        for v in neighbors:
            if v in outer_vertices:
                try:
                    bbb[i] += float(pos[v][0]) / n
                    ddd[i] += float(pos[v][1]) / n
                except:
                    # ipdb.set_trace()
                    print 'error with tutte embedding calculation'
                    badv = v


            else:
                j = remaining_vertices.index(v)
                AAA[i][j] = -(1 / float(n))
                CCC[i][j] = -(1 / float(n))
    x = np.linalg.solve(AAA, bbb)
    y = np.linalg.solve(CCC, ddd)
    for u in remaining_vertices:
        i = remaining_vertices.index(u)
        pos[u] = [x[i], y[i]]
    # ipdb.set_trace()
    # if len(graph) != len(pos):
    #     ipdb.set_trace()
    return pos


def barcode():
    base = 'ATCG'
    barcd = ''
    for i in range(0, 5):
        barcd += random.choice(base)
    return barcd

def uniq(lst):
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item
def sort_and_deduplicate(l):
    return list(uniq(sorted(l,reverse=True)))

def convert_vlines_to_edges(enumerated_faces,edgelist):
    # now convert vline IDs back into edges
    edges_faces_enumerated = []
    for face in range(0, len(enumerated_faces)):
        corresponding_face = []
        for line in range(0, len(enumerated_faces[face])):
            corresponding_edge = (edgelist[enumerated_faces[face][line]][0], edgelist[enumerated_faces[face][line]][1])
            corresponding_face += [corresponding_edge]
        edges_faces_enumerated += [corresponding_face]
    return edges_faces_enumerated

def get_vline_size_of_face(face_entry, all_vlines):
    vsizes_enumerated_faces = []
    for vline in range(0,len(face_entry)):
        vline_size = all_vlines[face_entry[vline]][2] - all_vlines[face_entry[vline]][1]
        vsizes_enumerated_faces += [vline_size]
    return vsizes_enumerated_faces

def sort_face_by_xposn(facetosort,all_vlines):
    x_positions = np.zeros(len(facetosort))
    for edge_to_sort in range(0, len(facetosort)):
        x_positions[edge_to_sort] = all_vlines[facetosort[edge_to_sort]][0]
    sorted_face = [x for _, x in sorted(zip(x_positions, facetosort))]
    return sorted_face
def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return array[idx-1]
    else:
        return array[idx]

def get_pixel_rgb(scaled_imarray, lookup_x_axis, lookup_y_axis, grid_x_lookup_value,grid_y_lookup_value):
    xindex = np.where(lookup_x_axis==find_nearest(lookup_x_axis,grid_x_lookup_value))[0][0]
    yindex = np.where(lookup_y_axis==find_nearest(lookup_y_axis,grid_y_lookup_value))[0][0]
    rgb = scaled_imarray[xindex,yindex]
    gray = np.average(scaled_imarray[xindex,yindex])
    return rgb, gray

# intput: a grpah as a dictionary
# output: a list of lists of vertices in the grpah which share neighbors
def same_neighbors(graph):
    same_neighbors = []
    for u in graph:
        same_neighbors_u = [u]
        for v in graph:
            if v != u:
                if set(graph[u]) == set(graph[v]):
                    same_neighbors_u.append(v)
        if len(same_neighbors_u) > 1:
            same_neighbors.append(same_neighbors_u)
    same_neighbors = [set(x) for x in same_neighbors]
    same = []
    for i in same_neighbors:
        if i not in same:
            same.append(i)
    same = [list(x) for x in same]
    return same

def grid(xlen,ylen,randomseed=4):
    #create a hexagonal lattice of points representing locations of oligos comprising the lawn
    np.random.seed(randomseed)
    num = xlen * ylen
    n= 2 * num
    points_1 = np.zeros((num, 2))
    points_2 = np.zeros((num, 2))
    for i in range(0, xlen):  # first grid
        for j in range(0, ylen):
            points_1[i * ylen + j, 0] = i
            points_1[i * ylen + j, 1] = (j * math.sqrt(3))
    for i in range(0, num):  # second grid
        points_2[i, 0] = points_1[i, 0] + 0.5
        points_2[i, 1] = points_1[i, 1] + math.sqrt(3) / 2
    points = np.vstack((points_1, points_2))  # paste together vertically
    np.savetxt(directory_name+'/'+'points', points, delimiter=",")
    return points#coordinates

def convert_dict_to_list(dict):
    dictList = []
    temp = []
    for key, value in dict.iteritems():
        temp = [value[0],value[1]]
        dictList.append(temp)
    return dictList

def rotate(x,y,xo,yo,theta): #rotate x,y around xo,yo by theta (rad)
    xr=math.cos(theta)*(x-xo)-math.sin(theta)*(y-yo)   + xo
    yr=math.sin(theta)*(x-xo)+math.cos(theta)*(y-yo)  + yo
    return [xr,yr]

def rotate_series(series,theta,originx,originy):
    rotated_series = np.zeros((len(series),2))
    for coordinate in range(0,len(series)):
        new_coordinate = rotate(series[coordinate][0], series[coordinate][1], originx, originy, theta)
        # ipdb.set_trace()
        rotated_series[coordinate][0] = new_coordinate[0]
        rotated_series[coordinate][1] = new_coordinate[1]
    return rotated_series

def get_radial_profile(points,distances,title='radial_error_plot',full_output=True):
    plt.close()
    binno = len(points)/10.
    ideal_radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    ideal_radii = ideal_radii / np.max(ideal_radii)
    bins = np.arange(0.0,1.0,1.0/binno)
    binned_sums = np.zeros((len(bins)))
    binned_counts = np.zeros((len(bins)))
    for bin in range(0,len(points)):
        binned_sums[searchify(bins,ideal_radii[bin])] += distances[bin]
        binned_counts[searchify(bins, ideal_radii[bin])] += 1

    binned_averages = binned_sums/binned_counts
    binned_averages[np.isnan(binned_averages)] = 0
    # ipdb.set_trace()
    np.savetxt(directory_name + '/' +'movingaverage'+title+'.txt',zip(bins,binned_averages))
    np.savetxt(directory_name + '/' +'radialerrors'+title+'.txt', zip(ideal_radii, distances))
    if full_output==True:
        plt.scatter(ideal_radii,distances,c='k',marker='o', alpha=0.5)
        plt.plot(bins, binned_averages,'r-')
        plt.xlabel('normalized radius')
        plt.ylabel('normalized polony displacement')
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.ylim(0,1.0)
        plt.savefig(directory_name+'/'+title+'.svg')
        plt.savefig(directory_name + '/' + title + '.png')
        plt.close()
    # ipdb.set_trace()
    return np.nanmean(distances)

def searchify(array, value):
    array = np.asarray(array)
    i = (np.abs(array - value)).argmin()
    return i

def evaluate_distortion(parameters, args):
    theta = parameters[0]
    x_translate = parameters[1]
    y_translate = parameters[2]
    scale = parameters[3]
    corseed, reconstructed, full_output,edge_number = args[0], args[1], args[2],args[3]
    master_error = 0
    #scale uniformly
    corseed = corseed*scale
    #translate all points to a new origin
    corseed[:,0] = corseed[:,0] + x_translate
    corseed[:, 1] = corseed[:, 1] + y_translate
    #rotate the reconstructed about the origin 0,0 since we have arrayed around unit circle
    corseed = rotate_series(corseed, theta,x_translate,y_translate)
    all_distances = np.zeros((len(reconstructed)))
    for polony in range(0,len(reconstructed)):
        # distance = np.sqrt((corseed[polony][0]-reconstructed[polony][0])**2+(corseed[polony][1]-reconstructed[polony][1])**2)
        try: distance = np.linalg.norm(corseed[polony]-reconstructed[polony])
        except:
            pass #ipdb.set_trace() #some kind of incomplete data error
        all_distances[polony] = distance
    mean_distance = np.average(all_distances)
    mean_cubed_distance = np.average(all_distances**3)
    stdev_distance = np.std(all_distances)
    skewness_distance = scipy.stats.skew(all_distances)
    master_error = (mean_cubed_distance*np.sqrt(len(reconstructed)))#*(np.abs(skewness_distance)+1) #this means a perfectly symmetric distance distribution would not alter the mean distance
    if full_output == False:
        return master_error
    else:
        return master_error,corseed

def evaluate_distortionbackup(parameters, args):
    theta = parameters[0]
    x_translate = parameters[1]
    y_translate = parameters[2]
    scale = parameters[3]
    corseed, reconstructed, full_output,edge_number = args[0], args[1], args[2],args[3]
    master_error = 0
    #scale uniformly
    corseed = corseed*scale
    #translate all points to a new origin
    corseed[:,0] = corseed[:,0] + x_translate
    corseed[:, 1] = corseed[:, 1] + y_translate
    #rotate the reconstructed about the origin 0,0 since we have arrayed around unit circle
    corseed = rotate_series(corseed, theta,x_translate,y_translate)
    all_distances = np.zeros((len(corseed)))
    for polony in range(0,len(corseed)):
        # distance = np.sqrt((corseed[polony][0]-reconstructed[polony][0])**2+(corseed[polony][1]-reconstructed[polony][1])**2)
        try: distance = np.linalg.norm(corseed[polony]-reconstructed[polony])
        except: ipdb.set_trace() #some kind of incomplete data error
        all_distances[polony] = distance
    mean_distance = np.average(all_distances)
    mean_cubed_distance = np.average(all_distances**3)
    stdev_distance = np.std(all_distances)
    skewness_distance = scipy.stats.skew(all_distances)
    master_error = (mean_cubed_distance*np.sqrt(len(corseed)))#*(np.abs(skewness_distance)+1) #this means a perfectly symmetric distance distribution would not alter the mean distance
    if full_output == False:
        return master_error
    else:
        return master_error,corseed

def reverse(seq):
    base = 'ATCG'
    rbase = 'TAGC'
    n = len(seq)
    rseq = ""
    for i in range(0, n):
        for j in range(0, 4):
            if seq[n - i - 1] == base[j]:
                rseq += rbase[j]
    return rseq

def perfect_hybrid(p2,p2_secondary,bc):
    npoints=len(p2)
    perfect_hybrid = ["" for x in range(0, npoints)]
    for i in range(0, npoints):
        perfect_hybrid[i] = p2[i] + bc[int(p2_secondary[i])]
    np.savetxt(directory_name+'/'+ 'perfect_hybrid', perfect_hybrid, delimiter=",", fmt="%s")
    print 'hybrid is sequence  '
    return perfect_hybrid

def hybrid(p1,p2,points,seed):
    npoints = len(points)
    np1 = len(p1)

    def reverse(seq):
        base = 'ATCG'
        rbase = 'TAGC'
        n = len(seq)
        rseq = ""
        for i in range(0, n):
            for j in range(0, 4):
                if seq[n - i - 1] == base[j]:
                    rseq += rbase[j]
        return rseq

    def weighted_choice(weights):
        totals = []
        running_total = 0
        for w in weights:
            running_total += w
            totals.append(running_total)
        rnd = random.random() * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i
        region = np.zeros((1))
        int_dis = np.zeros((npoints, npoints))
        p = np.zeros((npoints))
        delta = 1
        s = 0
        r = 5
        for i in range(0, npoints - 1):  # points-points distance
            for j in range(i + 1, npoints):
                if int_dis[i, j] == 0:
                    int_dis[i, j] = (
                        np.linalg.norm(points[i]-points[j])
                        # math.sqrt((points[i, 0] - points[j, 0]) ** 2 + (points[i, 1] - points[j, 1]) ** 2)
                    )
                    int_dis[j, i] = int_dis[i, j]
        end = time.time()


        start = time.time()
        from scipy.stats import norm, multivariate_normal
        hybrid = ["" for x in range(0, npoints)]
        trash = np.zeros((npoints))
        for i in range(0, npoints):
            if trash[i] == 0:
                mul_nor = multivariate_normal(mean=points[i], cov=[[delta, s], [s, delta]])
                for j in range(i + 1, npoints):
                    if 0 < int_dis[i, j] <= r and trash[j] == 0:
                        p[j] = mul_nor.pdf(points[j])
                x = weighted_choice(p)
                p = np.zeros((npoints))
                if x != None:
                    hybrid[i] = p2[i] + p2[x]
                    trash[x] = 1
                    hybrid[x] = p2[x] + p2[i]
                    plt.plot([points[i, 0], points[x, 0]], [points[i, 1], points[x, 1]])
                else:
                    hybrid[i] = ''
        np.savetxt(directory_name+'/'+ 'hybrid', hybrid, delimiter=",",fmt="%s")  # loadtxt will exclude empty strings automatically

def cell(points,ncell,seed,hybrid,xlen,ylen):
    npoints = len(points)
    nseed = len(seed)
    color_cycle = ['orange', 'green', 'blue', 'pink', 'cyan', 'gray']
    corcell = np.zeros((ncell, 2))
    choices = [i for i in range(0, npoints)]
    cell = np.random.choice(choices, ncell, replace=False)
    for i in range(0, ncell):
        n = cell[i]
        corcell[i] = points[n, :]

    np.savetxt(directory_name+'/'+ 'cell_cordinate', corcell, delimiter=',')
    voronoi = pytess.voronoi(corcell)
    voronoipolys = list(voronoi)


    for i in range(0, len(voronoipolys)):
        for j in range(0, len(voronoipolys[i][1])):

            voronoipolys[i][1][j] = list(voronoipolys[i][1][j])
            if voronoipolys[i][1][j][0] < 0:#set x coordinates lower limit
                voronoipolys[i][1][j][0] = 0
            if voronoipolys[i][1][j][0] > xlen:#set x coord upper limit
                voronoipolys[i][1][j][0] = xlen
            if voronoipolys[i][1][j][1] > ylen*math.sqrt(3):#set y
                voronoipolys[i][1][j][1] = ylen*math.sqrt(3)

            if voronoipolys[i][1][j][1] < 0:
                voronoipolys[i][1][j][1] = 0

    for i in range(0, ncell):
        plt.text(corcell[i, 0], corcell[i, 1], '%s' % i, alpha=1, fontweight='bold')

    for i in range(0, len(voronoipolys)):
        plt.fill(*zip(*voronoipolys[i][1]), alpha=0.2)
    plt.savefig(directory_name+'/'+'cell.svg')
    vor_polony = Voronoi(seed)
    voronoi_plot_2d(vor_polony)
    for i in range(0, nseed):
        plt.text(seed[i, 0], seed[i, 1], '%s' % i, alpha=0.5)

    plt.savefig(directory_name+'/'+ 'cell_mask.svg')
    near = np.zeros((npoints))
    dis = np.zeros((ncell))

    group_cell = list()
    group_receptor = list()
    for i in range(0, npoints):
        for j in range(0, ncell):
            # dis[j] = (math.sqrt((points[i, 0] - corcell[j, 0]) ** 2 + (points[i, 1] - corcell[j, 1]) ** 2))
            dis[j] = (np.linalg.norm(points[i]-corcell[j]))
        near[i] = np.argmin(dis)
    for i in range(0, ncell):
        subgroup = list()
        for j in range(0, npoints):
            if near[j] == i:
                subgroup.append(j)
        group_cell.append(subgroup)
    np.save(directory_name+'/'+ 'point_cell', group_cell)
    cell_barcode = list()
    for i in range(0, ncell):
        single_cell_barcode = list()
        for j in group_cell[i]:

                single_cell_barcode.append(hybrid[j])
        cell_barcode.append(single_cell_barcode)

    np.savetxt(directory_name+'/'+ 'cell_barcode', cell_barcode, delimiter=",", fmt="%s")
    np.save(directory_name+'/'+ 'cell_bc', cell_barcode)
    print'cell_points save  index of points belong to that cell'
    return group_cell

def get_ideal_graph(delaunay):
    G = nx.Graph()
    G.add_edges_from(np.column_stack((delaunay.simplices[:,0],delaunay.simplices[:,1])))
    G.add_edges_from(np.column_stack((delaunay.simplices[:,1],delaunay.simplices[:,-1])))
    G.add_edges_from(np.column_stack((delaunay.simplices[:,-1],delaunay.simplices[:,0])))
    return G

###########complete face traingulation ###########################

def edge_to_node(face_edge):
    face_vertices = list()
    for i in face_edge:
        face_vertices.append(i[0])
        face_vertices.append(i[1])
    face_vertices = set(face_vertices)
    return face_vertices

def centroid(face_node,node_pos):
    face_node_pos=[node_pos[i] for i in face_node]
    x_list=[vertex[0] for vertex in face_node_pos]
    y_list=[vertex[1] for vertex in face_node_pos]
    nvertex=len(face_node)
    x=sum(x_list)/nvertex
    y=sum(y_list)/nvertex
    return x,y

def vertex_CCW_sort(face_edge_list,node_pos): ##sort node incident to each face and sort face by number of nodes(decreasing)
    face_node_list=[]
    nnode=[]
    for i in face_edge_list:
        face_node=edge_to_node(i)
        face_node=list(face_node)
        nnode.append(len(face_node))
        x_centroid,y_centroid=centroid(face_node,node_pos)
        vector_list=list()

        for j in face_node:
            vector_list.append(((node_pos[j][1]-y_centroid),(node_pos[j][0]-x_centroid)))
        arcangle=[np.arctan2(vector[0],vector[1]) for vector in vector_list]
        idx=np.argsort(arcangle)
        new=[face_node[sorted] for sorted  in idx]
        face_node_list.append(new)
    outer_face=np.argmax(nnode)
    face_node_list.remove(face_node_list[outer_face])
    return face_node_list

def triangulation_fix(face_edge_list,node_pos):
    face_node_list=vertex_CCW_sort(face_edge_list,node_pos)
    new_edge=[]
    while len(face_node_list)>0:
        check=face_node_list.pop()
        if len(check)>3:
            new_edge.append((check[0],check[2]))
            face_node_list.insert(0,check[0:3])
            check.remove(check[1])
            face_node_list.insert(0, check)
        if face_node_list==None:
            break
    return new_edge

def alignment_prcs(bc,hy,cell_point):
    score = np.zeros((len(hy), len(bc)))
    times = np.zeros((len(bc)))
    best = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][0:5]:
                best[i] = j
                times[j] += 1
        if j == len(bc) and best[i] == 0:  ###############avoid p2 fail to pair resulting in polony0 false positive
            best[i] = None

    np.savetxt(directory_name+'/'+ 'best', best, delimiter=",")
    best_2 = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][5:10]:
                best_2[i] = j
                times[j] += 1
        if j == len(bc) and best_2[i] == 0:  ###############avoid p2 fail to pair resulting in polony0 false positive
            best_2[i] = None
    best_2 = np.zeros((len(hy)))
    for i in range(0, len(hy)):
        for j in range(0, len(bc)):
            if bc[j] == hy[i][5:10]:
                best_2[i] = j

    np.savetxt(directory_name+'/'+ 'best_r', best_2, delimiter=",")
    ajc_polony = np.zeros((len(bc), len(bc)))
    ajc_polony.astype(int)
    for i in range(0, len(hy)):
        if best[i] != None and best_2[i] != None:
            r = int(best[i])
            c = int(best_2[i])
            ajc_polony[r, c] = 1
            ajc_polony[c, r] = 1

    np.savetxt(directory_name+'/'+ 'ajc_polony', ajc_polony, delimiter=",", fmt="%i")
    cell_hybrid = list()
    for i in range(0, len(cell_point)):
        subgroup = list()
        for j in cell_point[i]:  ######it's easy to get lost in index and value as index
            subgroup.append(hy[j])
        cell_hybrid.append(subgroup)

    cell_polony = list()
    for i in range(0, len(cell_hybrid)):
        subgroup = list()
        for j in cell_hybrid[i]:
            for k in j:
                subgroup.append(j[0])

        subgroup = np.unique(subgroup)
        subgroup = subgroup[~np.isnan(subgroup)]
        cell_polony.append(subgroup)
    polony_cell = np.zeros(len(bc))
    for i in range(0, len(cell_polony)):
        for j in cell_polony[i]:
            polony_cell[int(j)] = i

def circular_layout_sorted(graph,radius):
    #takes a circular graph, sorts them, and arranges them uniformly around a circle, returns the positions

    edges_to_sort = list(graph.edges())
    all_nodes = list(graph.nodes())
    maxpath = []
    for node in range(0, len(all_nodes)):
        for other_node in range(0, len(all_nodes)):
            try: path = max(nx.all_simple_paths(graph,all_nodes[node],all_nodes[other_node]))
            except: path = nx.all_simple_paths(graph,all_nodes[node],all_nodes[other_node])
            try:
                path_size = len(path)
                if path_size > len(maxpath):
                    maxpath = path
            except: pass

    pos = dict()
    delta_theta = 2*np.pi / len(maxpath)
    for arc in range(0,len(maxpath)):
        node_to_place = maxpath[arc]
        pos_x = np.cos(delta_theta*arc)
        pos_y = np.sin(delta_theta*arc)
        pos.update({node_to_place:(pos_x,pos_y)})
    return pos

def get_euclidean_edgelength(edge,positions):
    point_a = edge[0]
    # ipdb.set_trace()
    a_x = positions[point_a][0]
    a_y = positions[point_a][1]
    point_b = edge[1]
    b_x = positions[point_b][0]
    b_y = positions[point_b][1]
    edge_length = np.sqrt((a_x-b_x)**2+(a_y-b_y)**2)
    return edge_length







def positiondict_to_list(pos_dict):
    flatlist = []
    dictlistx = []
    dictlisty = []
    for key, value in pos_dict.iteritems():
        temp = [value[0],value[1]]
        flatlist.append(temp)
        dictlistx.append(value[0])
        dictlisty.append(value[1])
    flatlist = [item for sublist in flatlist for item in sublist]
    return flatlist, dictlistx, dictlisty

def callback_on_optimization_indicator(x, f, accepted):
    if f == 0:
        print(x, f, accepted)
        return True
    else:
        return False

    # def alignbackup(self,title='alignment',error_threshold = 50,edge_number = 0, full_output = True,image_only=False):
    #     plt.close()
    #     minerror = 99999999
    #     if edge_number == 0:
    #         edge_number = len(self.corseed)
    #     minevals = 0
    #     while minerror > error_threshold and minevals < 1000:
    #         self.corseed = self.corseed * np.random.choice([-1.,1.],2) #generates a 2 item list of either -1s or +1s to flop coordinates
    #         initialize = np.array([random.uniform(0.,6.3),random.uniform(-10, 10.),random.uniform(-10, 10.),random.uniform(0.,1)])
    #         res1 = minimize(fun=evaluate_distortion, x0=initialize,#x0=np.array([.1, 6., 2., .9]),
    #                          args=([self.corseed, self.reconstructed_points, False,edge_number],), method='TNC', jac=None,
    #                          bounds=[(0., float('inf')),
    #                                  (float('-inf'), float('inf')),
    #                                  (float('-inf'), float('inf')),
    #                                  (.001, float('inf'))],
    #                         tol=1e-12,options={'eps': 1e-08,
    #                                            'scale': None,
    #                                            'offset': None,
    #                                            'mesg_num': None,
    #                                            'maxCGit': -1,
    #                                            'maxiter': 1000000,
    #                                            'eta': -1,
    #                                            'stepmx': 0,
    #                                            'accuracy': 0,
    #                                            'minfev': 1000,
    #                                            'ftol': -1,
    #                                            'xtol': -1,
    #                                            'gtol': -1,
    #                                            'rescale': -1,
    #                                            'disp': False}
    #                     )
    #         # alignment_minimizer_kwargs = {"tol":0.01,'method':'TNC',"args": ([corseed, reconstructed_points, False, edge_number],)}
    #         # res1 = scipy.optimize.basinhopping(func=evaluate_distortion,
    #         #                             x0=np.array([.1, 6., 2., .9]),
    #         #                             minimizer_kwargs=alignment_minimizer_kwargs,
    #         #                             niter=1000000,
    #         #                             disp=False,
    #         #                             niter_success=200,
    #         #                             callback=callback_on_optimization_indicator)
    #         if minerror > res1['fun']:
    #             minerror = res1['fun']
    #         else:
    #             pass
    #         minevals += 1
    #
    #     xopt1 = res1['x']
    #     error1, self.corseed_adjusted = evaluate_distortion(xopt1,args=[self.corseed, self.reconstructed_points,True,True,edge_number])
    #
    #     if full_output == True:
    #         plt.close()
    #         plt.scatter(self.corseed_adjusted[:,0],self.corseed_adjusted[:,1])
    #         plt.scatter(self.reconstructed_points[:,0],self.reconstructed_points[:,1])
    #     self.distances = []
    #     for i in range(0, len(self.reconstructed_points)):
    #         if full_output == True:
    #             plt.text(self.corseed_adjusted[i, 0], self.corseed_adjusted[i, 1], '%s' % i, alpha=0.5)
    #             plt.text(self.reconstructed_points[i, 0], self.reconstructed_points[i, 1], '%s' % i, alpha=0.5)
    #             plt.plot([self.corseed_adjusted[i, 0],self.reconstructed_points[i, 0]],[self.corseed_adjusted[i, 1],self.reconstructed_points[i, 1]],'k-')
    #         #self.distances += [np.sqrt((self.corseed_adjusted[i, 0]-self.reconstructed_points[i, 0])**2+(self.corseed_adjusted[i, 1]-self.reconstructed_points[i, 1])**2)]
    #         self.distances += [np.linalg.norm(self.cordseed_adjusted[i] - self.reconstructed_points[i])]
    #     if full_output == True:
    #         plt.rcParams['figure.figsize'] = (10, 10)
    #         plt.xlim(-1.1,1.1)
    #         plt.ylim(-1.1, 1.1)
    #         tri1 = Delaunay(self.corseed_adjusted)
    #         plt.triplot(self.corseed_adjusted[:, 0], self.corseed_adjusted[:, 1], tri1.simplices.copy(),'b-',alpha=0.2,marker='')
    #         if self.untethered_graph == 'None':
    #             tri2 = Delaunay(self.reconstructed_points)
    #             plt.triplot(self.reconstructed_points[:, 0], self.reconstructed_points[:, 1], tri2.simplices.copy(),'r-',alpha=0.2,marker='')
    #         else:
    #             nx.draw_networkx_edges(self.untethered_graph,self.reconstructed_points,width=1.0,alpha=0.2,edge_color='r')
    #             nx.draw_networkx_nodes(self.untethered_graph, self.reconstructed_points,node_color='r',node_size=25,alpha=0.2)
    #         plt.savefig(self.directory_name+'/'+ title +'redblue.svg')
    #         plt.savefig(self.directory_name + '/' + title + 'redblue.png')
    #         plt.close()
    #
    #     if full_output == True or image_only == True:
    #         plt.close()
    #         cmap = matplotlib.cm.get_cmap('YlOrRd')
    #         normalize = matplotlib.colors.Normalize(vmin=min(self.distances), vmax=max(self.distances))
    #         colors = [cmap(normalize(value)) for value in self.distances]
    #
    #         if len(self.corseed_adjusted) < 200:
    #             plt.scatter(self.corseed_adjusted[:,0],self.corseed_adjusted[:,1],c='k', alpha=0.1)
    #             plt.scatter(self.reconstructed_points[:,0],self.reconstructed_points[:,1],c='k',marker='o', alpha=0.1)
    #         for i in range(0, len(self.reconstructed_points)):
    #             plt.plot([self.corseed_adjusted[i, 0],self.reconstructed_points[i, 0]],[self.corseed_adjusted[i, 1],self.reconstructed_points[i, 1]],'-',color=colors[i])
    #         plt.rcParams['figure.figsize'] = (10, 10)
    #         plt.xlim(-1.1,1.1)
    #         plt.ylim(-1.1, 1.1)
    #         plt.savefig(self.directory_name+'/'+ title +'heatmap.svg')
    #         plt.savefig(self.directory_name + '/' + title + 'heatmap.png')
    #         plt.close()
    #         print 'final cost function value: ',minerror

    # def update_enumerated_faces(self, enumerated_faces, vsizes_enumerated_faces, partial_face,partial_face_vsize, all_vlines, edgelist,full_output=True):
    #     face_added = False
    #     # print 'CANDIDATE VLINE IDS:', partial_face, 'CANDIDATE SIZES: ', partial_face_vsize,'candidate edges: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    #     # if partial_face == [0,7] or partial_face == [0,4]:
    #     #     ipdb.set_trace()
    #     if len(enumerated_faces) == 0:
    #         enumerated_faces += [partial_face]
    #         vsizes_enumerated_faces += [get_vline_size_of_face(enumerated_faces[0], all_vlines)]
    #         face_added = True
    #         # print 'AN EMPTY ENUMERATED LIST WAS FOUND'
    #         if full_output==True:
    #             print 'FACE ADDED via empty start: : : : : ', partial_face, ' to ', enumerated_faces,'candidate edges: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    #     else:
    #         for face in range(0, len(enumerated_faces)):
    #             # ipdb.set_trace()
    #             # print 'FACE# ', face, 'vlines logged so far: ', enumerated_faces[face], 'and their sizes: ', vsizes_enumerated_faces[face]
    #             # above we sorted based on spatial order, face to the left or right of the vline in question
    #             # we organize the partially completed faces so that they are also spatially ordered
    #             # what we want to know is which side of the face is incomplete
    #             # every face has a large edge on one side, and multiple smaller edges on the other side
    #             # to merge a partial face with one of the existing partial faces, we must first rule out any that have ...
    #             # a different largest side
    #             # then we add only those vlines which are of the smaller side
    #             ####   check if the two partial faces have the same size symmetry ####
    #             if (vsizes_enumerated_faces[face][0] - vsizes_enumerated_faces[face][-1]) * (partial_face_vsize[0] - partial_face_vsize[1]) > 0:  # then they are of same sign
    #                 # print 'compatible symetries. size of enum first element, last, candidate first, last: ',vsizes_enumerated_faces[face], partial_face_vsize
    #                 # next we need to find out if the two faces are not just compatible in terms of size, but also if their largest
    #                 # edge is in fact the same edge. these two criteria, i.e. having the same largest edge and that it be on the same left/right side is sufficient to say
    #                 # that they are of the same face! some may have same largest edge but be on two different sides of that edge for example - being different faces
    #                 # we can use argmax of both partial faces in order to ignore having to check which side it's on
    #                 if enumerated_faces[face][np.argmax(vsizes_enumerated_faces[face])] == partial_face[np.argmax(partial_face_vsize)]:
    #                     # print 'largest edge match (enumed, candidate): ',enumerated_faces[face], partial_face
    #                     # now we have determined that the two faces are the same, we must then merge them except in the case where
    #                     # they are not just the same face, but the same set of edges - a redundant entry, which we can then skip
    #                     smaller_edge = partial_face[np.argmin(partial_face_vsize)]
    #                     if smaller_edge not in enumerated_faces[face]:
    #                         # print 'non-redundancy criterion ok (smaller edge of candidate, enumed face:', smaller_edge, enumerated_faces[face]
    #                         # now that we know that the two faces are same and that the entries are different, we can perform a merge action
    #                         # next use argmin of the partial face in question to send the un-added edge into the incomplete enumerated face
    #                         # while we're at it, we should sort the final entry by xpos
    #                         ###MERGE MERGE MERGE ###
    #
    #                         ### SORT SORT SORT ###
    #                         ##### check if this is an outer face first, if so then shift leftmost edge to the other end
    #
    #                         sorted_face = sort_face_by_xposn(facetosort=enumerated_faces[face],
    #                                     all_vlines=all_vlines)
    #                         sorted_after_merging = sort_face_by_xposn(facetosort=enumerated_faces[face]+[smaller_edge],
    #                                     all_vlines=all_vlines)
    #                         # if len(enumerated_faces[face])>2:
    #                         first_vlines_xposn = all_vlines[enumerated_faces[face][0]][0]
    #                         second_vlines_xposn = all_vlines[enumerated_faces[face][1]][0]
    #                         last_vlines_xposn = all_vlines[enumerated_faces[face][-1]][0]
    #                         secondtolast_vlines_xposn = all_vlines[enumerated_faces[face][-2]][0]
    #                         if second_vlines_xposn < first_vlines_xposn: #then this is an outer face, and after sorting we must shift stack
    #                             enumerated_faces[face] = [sorted_after_merging[-1]] + sorted_after_merging[0:-1]
    #                         elif last_vlines_xposn < secondtolast_vlines_xposn: #then this is the other type of outer face
    #                             enumerated_faces[face] = sorted_after_merging[1:]+ [sorted_after_merging[0]]
    #                         elif first_vlines_xposn < second_vlines_xposn and secondtolast_vlines_xposn < last_vlines_xposn:
    #                             #then the array appears to be sorted normally, meaning it is a face that does not span the periodic boundary
    #                             enumerated_faces[face] = sorted_after_merging
    #                         else:
    #                             if full_output == True:print 'error with face sorting during edge merge attempt'
    #                             ipdb.set_trace()
    #                         # else:
    #                         #     enumerated_faces[face] = sorted_after_merging
    #
    #
    #
    #                         vsizes_enumerated_faces[face] = get_vline_size_of_face(enumerated_faces[face],all_vlines)
    #                         face_added = True
    #                         if full_output == True:print 'FACE ADDED via merge event : : : : : ', partial_face, ' to ', enumerated_faces[face], 'edges are ', convert_vlines_to_edges(enumerated_faces, edgelist)[face],'candidate edges: ', edgelist[partial_face[0]], ' and ', edgelist[partial_face[1]]
    #                         break
    #                     else:
    #                         # print 'REJECT we found a redundant face: ', enumerated_faces[face], ' and ', partial_face
    #                         # this must be a redundant face, we should not readd it to the archive
    #                         face_added = True
    #                         break
    #                 else:
    #                     pass# print 'REJECT - despite having same spatial class, the largest edge was not the same edge for the two faces: ', enumerated_faces[face][np.argmax(vsizes_enumerated_faces[face])], ' and ',  partial_face[np.argmax(partial_face_vsize)]
    #             else:
    #                 pass # print 'REJECT - incompatible symmetries: ', vsizes_enumerated_faces[face], ' vs ', partial_face_vsize
    #     # now that we've checked for this very special case of a face which shares both same largest edge and same
    #     # left-right symmetry of the largest edge and finally accounted for redundant entries and merged such a case
    #     # if we made it through without triggering either of the face-added status
    #     # then we can simply add the partial entry as a new incomplete face to the list of enumerated faces
    #     # it will then be revisited for additions until completed
    #     if face_added == False:
    #         enumerated_faces += [partial_face]
    #         vsizes_enumerated_faces += [get_vline_size_of_face(enumerated_faces[-1], all_vlines)]
    #         face_added = True
    #         if full_output == True:print 'FACE ADDED via last resort : : : : : ', partial_face, ' to ', enumerated_faces,'candidate edges added: ', edgelist[partial_face[0]][0:2], ' and ', edgelist[partial_face[1]][0:2]
    #     face_edge_list = convert_vlines_to_edges(enumerated_faces, edgelist)
    #     # ipdb.set_trace()
    #     return enumerated_faces, vsizes_enumerated_faces

    # def planar_embedding_draw(self, graph, labels=True, title='',full_output=True):
    #     pgraph = planarity.PGraph(graph)
    #     pgraph.embed_drawplanar()
    #     hgraph = planarity.networkx_graph(pgraph) #returns a networkx graph built from planar graph- nodes and edges only
    #     patches = []
    #     node_labels = {}
    #     xs = []
    #     ys = []
    #     all_vlines = []
    #     xmax=0
    #     for node, data in hgraph.nodes(data=True):
    #         y = data['pos']
    #         xb = data['start']
    #         xe = data['end']
    #         x = int((xe+xb)/2)
    #         node_labels[node] = (x, y)
    #         patches += [Circle((x, y), 0.25)]#,0.5,fc='w')]
    #         xs.extend([xb, xe])
    #         ys.append(y)
    #         plt.hlines([y], [xb], [xe])
    #     edgelist = list(hgraph.edges(data=True)) #only
    #
    #     for i in range(0,len(edgelist)):
    #         #print i
    #         data = edgelist[i][2]
    #         x = data['pos']
    #         if x > xmax: xmax = x
    #         yb = data['start']
    #         ye = data['end']
    #         ys.extend([yb, ye])
    #         xs.append(x)
    #         all_vlines += [[x, yb,ye]]
    #         plt.vlines([x], [yb], [ye])
    #     # labels
    #     if labels:
    #         for n, (x, y) in node_labels.items():
    #             plt.text(x, y, n,
    #                      horizontalalignment='center',
    #                      verticalalignment='center',
    #                      bbox = dict(boxstyle='round',
    #                                  ec=(0.0, 0.0, 0.0),
    #                                  fc=(1.0, 1.0, 1.0),
    #                                  )
    #                      )
    #     p = PatchCollection(patches)
    #     if full_output == True:
    #         ax = plt.gca()
    #         ax.add_collection(p)
    #         plt.axis('equal')
    #         plt.xlim(min(xs)-1, max(xs)+1)
    #         plt.ylim(min(ys)-1, max(ys)+1)
    #         plt.savefig(self.directory_name + '/' +'planar_embedding_diagram'+title+'.svg')
    #         plt.savefig(self.directory_name + '/' + 'planar_embedding_diagram' + title + '.png')
    #         plt.close()
    #     xmax += 1
    #     enumerated_faces = []
    #     vsizes_enumerated_faces = []
    #     for vline in range(0,len(all_vlines)):
    #         if full_output == True: print vline
    #         #find closest interesting vline for top part, and bottom part resp.
    #         minimum_positive_distance = 999999999999
    #         edge_candidate_positive = []
    #         minimum_negative_distance = -999999999999
    #         edge_candidate_negative = []
    #         edge_candidate_primary = [edgelist[vline][0],edgelist[vline][1],edgelist[vline][2]['pos']]
    #         #intersectability - we create an intersectors group,
    #         #ie the set of vlines that have vertical overlap with our current vline
    #         #start by making an empty intersectors list that we will add to below.
    #         # ultimately all we want from these are the minima, the vlines closest to our vline with the potential to ...
    #         # ...intersect, these will be the starts of faces
    #         #note that each intersector has two associated lateral distances, from left and right, negative and positive
    #         #this means we enforce a periodic boundary condition, so in the case of othervline being to the right of vline,
    #         #then we would add the distance from the leftmost edge to vline with the distance from rightmost edge to other_vline
    #         intersectors = [[],[],[]] #[[identifying index of other_vline],[distance btw vline and other_vline to the left],[" for distance via the right]]
    #         # ipdb.set_trace()
    #         for other_vline in range(0, len(all_vlines)):
    #             #check vertical intersectability
    #             if all_vlines[vline][1] >= all_vlines[other_vline][2] or all_vlines[vline][2] <= all_vlines[other_vline][1] or all_vlines[vline][0] == all_vlines[other_vline][0]:
    #                 pass
    #             else: #if vertical overlap exists, then we need to add the other_vline to the list of intersectors of vline
    #                 intersectors[0] += [other_vline]
    #                 distance_to_right = (all_vlines[vline][0] - all_vlines[other_vline][0])%xmax
    #                 distance_to_left = (all_vlines[other_vline][0] - all_vlines[vline][0])%xmax
    #                 intersectors[2] += [distance_to_left]
    #                 intersectors[1] += [distance_to_right]
    #         closest_to_left = intersectors[0][np.argmin(intersectors[1])]
    #         partial_face_to_left = [closest_to_left,vline]
    #         partial_face_to_left_vsize = [all_vlines[closest_to_left][2]-all_vlines[closest_to_left][1],all_vlines[vline][2]-all_vlines[vline][1]]
    #         closest_to_right = intersectors[0][np.argmin(intersectors[2])]
    #         partial_face_to_right = [vline,closest_to_right]
    #         partial_face_to_right_vsize = [all_vlines[vline][2]-all_vlines[vline][1],all_vlines[closest_to_right][2]-all_vlines[closest_to_right][1]]
    #         enumerated_faces,vsizes_enumerated_faces = self.update_enumerated_faces(enumerated_faces, vsizes_enumerated_faces, partial_face_to_left, partial_face_to_left_vsize, all_vlines, edgelist,full_output=full_output)
    #         enumerated_faces,vsizes_enumerated_faces = self.update_enumerated_faces(enumerated_faces, vsizes_enumerated_faces, partial_face_to_right, partial_face_to_right_vsize, all_vlines,edgelist,full_output=full_output)
    #         face_edge_list = convert_vlines_to_edges(enumerated_faces,edgelist)
    #         # print        "END OF common face insertion block"
    #     # print 'END of vline loop'
    #     if full_output==True: 'End of Face Enumeration'
    #     ipdb.set_trace()
    #     return face_edge_list

