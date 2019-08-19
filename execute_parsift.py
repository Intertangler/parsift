import numpy as np
import sys
import ipdb
import matplotlib
import matplotlib.pyplot as plt
import random
import os
import parsift_lib
from select import select
import time
from inspect import currentframe, getframeinfo

class Experiment:
    def __init__(self, directory_label = ''):
        self.directory_label = directory_label

    def ideal_reconstruct(self,target_nsites=300, npolony=50, filename='monalisacolor.png', master_directory='', directory_label='', randomseed=0, iterlabel='',full_output=False, do_ideal_tutte = True):
        # reload(parsift_lib)
        # directory_name = master_directory + '/' + directory_label
        if not master_directory:
            directory_name = parsift_lib.prefix(directory_label)
        else:
            directory_name = master_directory

        ########## SURFACE SIMULATION STEPS ##########
        t0 = time.time()
        self.basic_circle = parsift_lib.Surface(target_nsites=target_nsites, directory_label='',
                                           master_directory=directory_name)
        self.basic_circle.circular_grid_ian()
        self.basic_circle.scale_image_axes(filename=filename)
        print 'begin seeding'
        self.basic_circle.seed(nseed=npolony, full_output=full_output)
        print 'begin crosslinking polonies'
        self.basic_circle.ideal_only_crosslink(full_output=full_output)
        surface_prep_t = time.time() - t0

        if do_ideal_tutte == True:
            print '########## TUTTE RECONSTRUCTION ##########'
            tuttet0 = time.time()
            self.tutte_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=None,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            self.tutte_reconstruction.conduct_tutte_embedding_from_ideal(full_output=full_output, image_only=False)

    def reconstruct(self,target_nsites=300, npolony=50, filename='monalisacolor.png', master_directory='',
                                   directory_label='', randomseed=0, iterlabel='',full_output=False, do_peripheral_pca = True,do_total_pca = True,do_spring = True,do_tutte = True,get_average_distortogram=False,optimal_linking=True,bipartite=False):
        # reload(parsift_lib)
        # directory_name = master_directory + '/' + directory_label
        if not master_directory:
            directory_name = parsift_lib.prefix(directory_label)
        else:
            directory_name = master_directory

        ########## SURFACE SIMULATION STEPS ##########
        t0 = time.time()
        self.basic_circle = parsift_lib.Surface(target_nsites=target_nsites, directory_label='',
                                           master_directory=directory_name)
        self.basic_circle.circular_grid_ian()
        self.basic_circle.scale_image_axes(filename=filename)
        print 'begin seeding'
        self.basic_circle.seed(nseed=npolony, full_output=full_output)
        print 'begin crosslinking polonies'
        self.basic_circle.crosslink_polonies(full_output=full_output,optimal_linking=optimal_linking,bipartite=bipartite)
        surface_prep_t = time.time() - t0
        self.basic_circle.check_continuity()

        if optimal_linking == False:
            self.n_crosspairs = len(self.basic_circle.cross_pairing_events)
            self.n_selfpairs = len(self.basic_circle.self_pairing_events)

        if do_tutte == True:
            print '########## TUTTE RECONSTRUCTION ##########'
            self.tutte_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            if self.basic_circle.disconnected == False:
                tuttet0 = time.time()
                self.tutte_reconstruction.conduct_tutte_embedding(full_output=full_output, image_only=False)


                #tutte_reconstruction.face_enumeration_runtime
                self.tuttetime = time.time() - tuttet0
                self.tutte_reconstruction.align(title='align_tutte_points' + iterlabel, error_threshold=1, full_output=full_output,get_average_distortogram=get_average_distortogram)
                self.tutte_reconstruction.get_delaunay_comparison(
                     parsift_lib.positiondict_to_list(self.tutte_reconstruction.reconstructed_pos)[0],
                     self.basic_circle.untethered_graph)
                self.tutte_err = parsift_lib.get_radial_profile(self.tutte_reconstruction.reconstructed_points,
                                                            self.tutte_reconstruction.distances, title='rad_tutte' + iterlabel)
                # self.tutte_reconstruction.conduct_tutte_embedding_from_ideal(full_output=full_output, image_only=False)

        if do_peripheral_pca == True:
            print '########## PERIPHERAL PCA RECONSTRUCTION ##########'
            self.peripheralpca_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)


            if self.basic_circle.disconnected == False:
                peripheralpca_time0 = time.time()
                self.peripheralpca_reconstruction.conduct_pca_peripheral(full_output=full_output, image_only=False)
                self.peripheralpca_time = time.time() - peripheralpca_time0
                self.peripheralpca_reconstruction.align(title='align_peripheralpca_points' + iterlabel, error_threshold=1, full_output=full_output,get_average_distortogram=get_average_distortogram)
                self.peripheralpca_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(self.peripheralpca_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)
                self.peripheralpca_err = parsift_lib.get_radial_profile(self.peripheralpca_reconstruction.reconstructed_points, self.peripheralpca_reconstruction.distances, title='rad_peripheralpca' + iterlabel)



        if do_total_pca == True:
            print '########## TOTAL PCA RECONSTRUCTION ##########'
            self.totalpca_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)

            if self.basic_circle.disconnected == False:
                totalpca_time0 = time.time()
                self.totalpca_reconstruction.conduct_shortest_path_matrix(full_output=full_output, image_only=False)


                self.totalpca_time = time.time() - totalpca_time0
                self.totalpca_reconstruction.align(title='align_totalpca_points' + iterlabel, error_threshold=1, full_output=full_output,get_average_distortogram=get_average_distortogram)
                self.totalpca_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(self.totalpca_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)
                self.totalpca_err = parsift_lib.get_radial_profile(self.totalpca_reconstruction.reconstructed_points,
                                                             self.totalpca_reconstruction.distances, title='rad_totalpca' + iterlabel)

        if do_spring == True:
            print '########## SPRING RECONSTRUCTION ##########'
            self.spring_reconstruction = parsift_lib.Reconstruction(self.basic_circle.directory_name,
                                                               corseed=self.basic_circle.corseed,
                                                               ideal_graph=self.basic_circle.ideal_graph,
                                                               untethered_graph=self.basic_circle.untethered_graph,
                                                               rgb_stamp_catalog=self.basic_circle.rgb_stamp_catalog)
            if self.basic_circle.disconnected == False:
                springt0 = time.time()
                self.spring_reconstruction.conduct_spring_embedding(full_output=full_output, image_only=False)

                # ipdb.set_trace()
                self.springtime = time.time() - springt0

                self.spring_reconstruction.align(title='align_spring_points' + iterlabel, error_threshold=1, full_output=full_output,get_average_distortogram=get_average_distortogram)
                self.spring_reconstruction.get_delaunay_comparison(
                    parsift_lib.positiondict_to_list(self.spring_reconstruction.reconstructed_pos)[0], self.basic_circle.untethered_graph)

                self.spring_err = parsift_lib.get_radial_profile(self.spring_reconstruction.reconstructed_points,
                                                            self.spring_reconstruction.distances, title='rad_spring' + iterlabel)








def ideal_experiment(target_nsites=20000,npolony=200):
    reload(parsift_lib)
    start_t = time.time()
    basic_run = Experiment(directory_label='MEGArun')
    basic_run.ideal_reconstruct(target_nsites=target_nsites, npolony=npolony, full_output=True, do_ideal_tutte=True,filename='comb.png')
    run_time = time.time() - start_t
    print run_time


#basic run with alignment
def basic_single_experiment_with_alignment(target_nsites=50*2000,npolony=50):
    reload(parsift_lib)
    start_t = time.time()
    basic_run = Experiment(directory_label='single')
    basic_run.reconstruct(target_nsites=target_nsites, npolony=npolony, full_output=True, do_peripheral_pca=False, do_total_pca=False,
                          do_spring=False, do_tutte=True,optimal_linking=False)
    run_time = time.time() - start_t
    print run_time

def basic_single_experiment_BIPARTITE(target_nsites=20*25,npolony=20):
    reload(parsift_lib)
    start_t = time.time()
    basic_run = Experiment(directory_label='single')
    basic_run.reconstruct(target_nsites=target_nsites, npolony=npolony, full_output=True, do_peripheral_pca=True, do_total_pca=True,
                          do_spring=True, do_tutte=True,optimal_linking=True,bipartite=True)
    run_time = time.time() - start_t
    print run_time

# def repeated_attempts_to_image_imperfection():
#
#     success = False
#     while success == False:
#         print 'attempting to form image'
#         try:
#             basic_single_experiment_with_alignment(target_nsites=500*20,npolony=500)
#         except:
#             pass


def mega_run_attempts():
    try: basic_single_experiment_with_alignment(target_nsites=240000,npolony=12000)
    except:
        try:
            basic_single_experiment_with_alignment(target_nsites=240000, npolony=12000)
        except:
            try:
                basic_single_experiment_with_alignment(target_nsites=240000, npolony=12000)
            except:
                try:
                    basic_single_experiment_with_alignment(target_nsites=240000, npolony=12000)
                except:
                    pass

    try: basic_single_experiment_with_alignment(target_nsites=800000,npolony=20000)
    except:
        try:
            basic_single_experiment_with_alignment(target_nsites=800000, npolony=20000)
        except:
            try:
                basic_single_experiment_with_alignment(target_nsites=800000, npolony=20000)
            except:
                try:
                    basic_single_experiment_with_alignment(target_nsites=800000, npolony=20000)
                except:
                    pass

def average_distortogram(min = 10,max = 20,step = 10,title = 'average_distortogram_1000x10_9spp_coreseedmap',repeats_per_step = 5,spp=500):
    # min = 10
    # max = 20
    # step = 10
    # title = 'average_distortogram_1000x10_9spp_coreseedmap'
    # repeats_per_step = 100
    # spp=24
    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    do_peripheral_pca = True
    do_total_pca = True
    do_spring = True
    do_tutte = True
    optimal_linking = False

    discretized_distortogram_dimension = 25

# if do_tutte == True:
    discretized_avg_distortogramtutte = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
    discretized_avg_distortogram_countertutte = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
# if do_spring == True:
    discretized_avg_distortogramspring = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
    discretized_avg_distortogram_counterspring = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
# if do_total_pca == True:
    discretized_avg_distortogramtotalpca = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
    discretized_avg_distortogram_countertotalpca = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
# if do_peripheral_pca == True:
    discretized_avg_distortogramperipheralpca = np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))
    discretized_avg_distortogram_counterperipheralpca= np.zeros((discretized_distortogram_dimension, discretized_distortogram_dimension))

    # for i in range(0, number_of_steps * repeats_per_step):
    i = 0
    while i < number_of_steps * repeats_per_step:
        try:
            print 'avg distortogram ',i, ' out of ', number_of_steps * repeats_per_step
            # try:
            npolony = min + i / repeats_per_step * step
            nsites = (min + i / repeats_per_step * step) * spp
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            # master_errors_spring[i], master_errors_tutte[i]
            # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            #     i], spring_run_times[i], tutte_run_times[i],face_enumeration_run_times[i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
            #                                     filename='monalisacolor.png', master_directory=master_directory,
            #                                     iterlabel=str(npolony))

            basic_run = Experiment(directory_label='single')
            basic_run.reconstruct(target_nsites=nsites, npolony=npolony,randomseed=i % repeats_per_step,filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(npolony), full_output=False, do_peripheral_pca=do_peripheral_pca, do_total_pca=do_total_pca,
                          do_spring=do_spring, do_tutte=do_tutte,get_average_distortogram=True, optimal_linking=optimal_linking)
            if do_tutte == True:
                discretized_avg_distortogramtutte += basic_run.tutte_reconstruction.discretized_distortogram
                discretized_avg_distortogram_countertutte += basic_run.tutte_reconstruction.discretized_distortogram_counter
            if do_spring == True:
                discretized_avg_distortogramspring += basic_run.spring_reconstruction.discretized_distortogram
                discretized_avg_distortogram_counterspring += basic_run.spring_reconstruction.discretized_distortogram_counter
            if do_total_pca == True:
                discretized_avg_distortogramtotalpca += basic_run.totalpca_reconstruction.discretized_distortogram
                discretized_avg_distortogram_countertotalpca += basic_run.totalpca_reconstruction.discretized_distortogram_counter
            if do_peripheral_pca == True:
                discretized_avg_distortogramperipheralpca += basic_run.peripheralpca_reconstruction.discretized_distortogram
                discretized_avg_distortogram_counterperipheralpca += basic_run.peripheralpca_reconstruction.discretized_distortogram_counter
            i += 1
        except:pass
        # except:
        #     print 'ERROR'
            # ipdb.set_trace()
        #     pass
    if do_tutte == True:
        discretized_avg_distortogramtutte = np.nan_to_num(discretized_avg_distortogramtutte/discretized_avg_distortogram_countertutte)
        # ipdb.set_trace()
        plt.imshow(discretized_avg_distortogramtutte,cmap='magma_r', interpolation="nearest")
        plt.clim(0,2)
        plt.axis('off')
        plt.savefig(master_directory + '/' + title + '_discretizedtutte.svg')
        plt.savefig(master_directory + '/' + title + '_discretizedtutte.png')
        plt.close()
    if do_spring == True:
        discretized_avg_distortogramspring = np.nan_to_num(discretized_avg_distortogramspring/discretized_avg_distortogram_counterspring)
        # ipdb.set_trace()
        plt.imshow(discretized_avg_distortogramspring,cmap='magma_r', interpolation="nearest")
        plt.clim(0,2)
        plt.axis('off')
        plt.savefig(master_directory + '/' + title + '_discretizedspring.svg')
        plt.savefig(master_directory + '/' + title + '_discretizedspring.png')
        plt.close()
    if do_total_pca == True:
        discretized_avg_distortogramtotalpca = np.nan_to_num(discretized_avg_distortogramtotalpca/discretized_avg_distortogram_countertotalpca)
        # ipdb.set_trace()
        plt.imshow(discretized_avg_distortogramtotalpca,cmap='magma_r', interpolation="nearest")
        plt.clim(0,2)
        plt.axis('off')
        plt.savefig(master_directory + '/' + title + '_discretizedtotalpca.svg')
        plt.savefig(master_directory + '/' + title + '_discretizedtotalpca.png')
        plt.close()
    if do_peripheral_pca == True:
        discretized_avg_distortogramperipheralpca = np.nan_to_num(discretized_avg_distortogramperipheralpca/discretized_avg_distortogram_counterperipheralpca)
        # ipdb.set_trace()
        plt.imshow(discretized_avg_distortogramperipheralpca,cmap='magma_r', interpolation="nearest")
        plt.clim(0,2)
        plt.axis('off')
        plt.savefig(master_directory + '/' + title + '_discretizedperipheralpca.svg')
        plt.savefig(master_directory + '/' + title + '_discretizedperipheralpca.png')
        plt.close()

    # timeout = 3
    # print 'cd to ' + master_directory + '? (Y/n)'
    # rlist, _, _ = select([sys.stdin], [], [], timeout)
    # if rlist:
    #     cd_query = sys.stdin.readline()
    #     # ipdb.set_trace()
    #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
    #     if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
    #         os.chdir(master_directory)
    #     else:
    #         print 'no cd'
    # else:
    #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
    #     os.chdir(master_directory)

def multiple_average_distortograms_site():
    for spp in [500,250,125,63,33]:
        average_distortogram(min=500, max=510, step=10, title='aAaAaaverage_distortogram_pol100_rep100_spp'+str(spp), repeats_per_step=10, spp=spp)
def multiple_average_distortograms_pol():
    for pol in [125,250,500]:
        average_distortogram(min=pol, max=pol+10, step=10, title='bBbBbaverage_distortogram_pol'+str(pol)+'_rep100_spp800', repeats_per_step=5000/pol, spp=1000)

def polony_number_variation_experiment():
    min = 50
    max = 150
    step = 50
    title = 'polony_variation'
    repeats_per_step = 25
    spp = 500

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    polony_counts = np.zeros((number_of_steps * repeats_per_step))


    master_errors_peripheralpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_totalpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))

    master_levenshtein_peripheralpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_totalpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step, 1))


    peripheralpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    totalpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    spring_run_times = np.zeros((number_of_steps * repeats_per_step))
    tutte_run_times = np.zeros((number_of_steps * repeats_per_step))
    face_enumeration_run_times = np.zeros((number_of_steps * repeats_per_step))

    i = 0
    while i < number_of_steps * repeats_per_step:
        try:
            print 'pol var ', i, ' out of ', number_of_steps * repeats_per_step
            npolony = min + i / repeats_per_step * step
            nsites = (min + i / repeats_per_step * step) * spp
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            # master_errors_spring[i], master_errors_tutte[i]
            # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            #     i], spring_run_times[i], tutte_run_times[i],face_enumeration_run_times[i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
            #                                     filename='monalisacolor.png', master_directory=master_directory,
            #                                     iterlabel=str(npolony))

            basic_run = Experiment(directory_label='single')
            basic_run.reconstruct(target_nsites=nsites, npolony=npolony,randomseed=i % repeats_per_step,filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(npolony), full_output=False, do_peripheral_pca=True, do_total_pca=True,
                          do_spring=True, do_tutte=True,optimal_linking=False)

            try:
                master_errors_peripheralpca[i] = basic_run.peripheralpca_err
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_errors_totalpca[i] = basic_run.totalpca_err
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_errors_spring[i] = basic_run.spring_err
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_errors_tutte[i] = basic_run.tutte_err
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno

            try:peripheralpca_run_times[i] = basic_run.peripheralpca_time
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:totalpca_run_times[i] = basic_run.totalpca_time
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:spring_run_times[i] = basic_run.springtime
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:tutte_run_times[i] = basic_run.tuttetime
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:face_enumeration_run_times[i] = basic_run.tutte_reconstruction.face_enumeration_runtime
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno

            try:master_levenshtein_peripheralpca[i] = basic_run.peripheralpca_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_levenshtein_totalpca[i] = basic_run.totalpca_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_levenshtein_spring[i] = basic_run.spring_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:master_levenshtein_tutte[i] = basic_run.tutte_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            # ipdb.set_trace()


            try:np.savetxt(master_directory + '/' + 'master_levenshtein_peripheralpca' + title + '.txt', zip(polony_counts, master_levenshtein_peripheralpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_totalpca' + title + '.txt', zip(polony_counts, master_levenshtein_totalpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_spring' + title + '.txt', zip(polony_counts, master_levenshtein_spring))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + 'master_levenshtein_tutte' + title + '.txt', zip(polony_counts, master_levenshtein_tutte))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno

            try:np.savetxt(master_directory + '/' + '0mastererrorsperipheralpca' + title + '.txt', zip(polony_counts, master_errors_peripheralpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '0mastererrorstotalpca' + title + '.txt', zip(polony_counts, master_errors_totalpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '0mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '0mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno

            try:np.savetxt(master_directory + '/' + '1peripheralpca_runtimes_' + title + '.txt', zip(polony_counts, peripheralpca_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '1totalpca_runtimes_' + title + '.txt', zip(polony_counts, totalpca_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '1spring_runtimes_' + title + '.txt', zip(polony_counts, spring_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            try:np.savetxt(master_directory + '/' + '1tutte_runtimes_' + title + '.txt', zip(polony_counts, tutte_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno

            try:np.savetxt(master_directory + '/' + '1face_enumeration_runtimes_' + title + '.txt', zip(polony_counts, face_enumeration_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename,frameinfo.lineno
            i += 1
        except:
            print 'error with this attempt'
            pass

    timeout = 3
    print 'cd to ' + master_directory + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
            os.chdir(master_directory)
        else:
            print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(master_directory)

def site_number_variation_experiment():
    npol = 10
    min = npol*50
    max = npol*50 + npol*50*50
    step = npol*10
    title = 'CCCsite_variation_10pol_50spp_increment_rep100'
    repeats_per_step = 25
    do_peripheral_pca = True
    do_total_pca = True
    do_spring = True
    do_tutte = True
    optimal_linking = False

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    itercounts = np.zeros((number_of_steps * repeats_per_step))


    master_errors_peripheralpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_totalpca = np.zeros((number_of_steps * repeats_per_step))
    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))

    master_levenshtein_peripheralpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_totalpca = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step, 1))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step, 1))


    peripheralpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    totalpca_run_times = np.zeros((number_of_steps * repeats_per_step))
    spring_run_times = np.zeros((number_of_steps * repeats_per_step))
    tutte_run_times = np.zeros((number_of_steps * repeats_per_step))
    face_enumeration_run_times = np.zeros((number_of_steps * repeats_per_step))

    linking_values = np.zeros((number_of_steps * repeats_per_step, 4))
    # ipdb.set_trace()
    # for i in range(0, number_of_steps * repeats_per_step):
        # try:
    i = 0
    while i < number_of_steps * repeats_per_step:
        try:
            nsites = min + i / repeats_per_step * step
            # substep = i%3
            print nsites
            print 'site var', i, ' out of ',  number_of_steps * repeats_per_step
            itercounts[i] = nsites
            # master_errors_spring[i], master_errors_tutte[i]
            # try:
                # basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
            #     i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=50, randomseed=i % repeats_per_step,
            #                                     filename='monalisacolor.png', master_directory=master_directory,
            #                                     iterlabel=str(nsites))
            basic_run = Experiment(directory_label='single')
            basic_run.reconstruct(target_nsites=nsites, npolony=npol,randomseed=i % repeats_per_step,filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(nsites), full_output=False, do_peripheral_pca=do_peripheral_pca, do_total_pca=do_total_pca,
                      do_spring=do_spring, do_tutte=do_tutte,optimal_linking=optimal_linking)


            if optimal_linking==False:
                linking_values[i][0] = basic_run.n_crosspairs
                linking_values[i][1] = basic_run.n_selfpairs
                linking_values[i][2] = nsites
                linking_values[i][3] = npol
                np.savetxt(master_directory + '/' + 'linking_counts' + title + '.txt', linking_values)
            if do_peripheral_pca==True:
                master_errors_peripheralpca[i] = basic_run.peripheralpca_err
                peripheralpca_run_times[i] = basic_run.peripheralpca_time
            # except:
            #     frameinfo = getframeinfo(currentframe())
            #     print frameinfo.filename, frameinfo.lineno
            # try:
            if do_total_pca == True:
                master_errors_totalpca[i] = basic_run.totalpca_err
                totalpca_run_times[i] = basic_run.totalpca_time
            # except:
            #     frameinfo = getframeinfo(currentframe())
            #     print frameinfo.filename, frameinfo.lineno
            # try:
            if do_spring == True:
                master_errors_spring[i] = basic_run.spring_err
                spring_run_times[i] = basic_run.springtime
            # except:
            #     frameinfo = getframeinfo(currentframe())
            #     print frameinfo.filename, frameinfo.lineno
            # try:
            if do_tutte == True:
                master_errors_tutte[i] = basic_run.tutte_err
                tutte_run_times[i] = basic_run.tuttetime
            try:
                face_enumeration_run_times[i] = basic_run.tutte_reconstruction.face_enumeration_runtime
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno

            try:
                master_levenshtein_peripheralpca[i] = basic_run.peripheralpca_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                master_levenshtein_totalpca[i] = basic_run.totalpca_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                master_levenshtein_spring[i] = basic_run.spring_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                master_levenshtein_tutte[i] = basic_run.tutte_reconstruction.levenshtein_distance
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            # ipdb.set_trace()

            try:
                np.savetxt(master_directory + '/' + 'master_levenshtein_peripheralpca' + title + '.txt',
                           zip(itercounts, master_levenshtein_peripheralpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + 'master_levenshtein_totalpca' + title + '.txt',
                           zip(itercounts, master_levenshtein_totalpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + 'master_levenshtein_spring' + title + '.txt',
                           zip(itercounts, master_levenshtein_spring))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + 'master_levenshtein_tutte' + title + '.txt',
                           zip(itercounts, master_levenshtein_tutte))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno

            try:
                np.savetxt(master_directory + '/' + '0mastererrorsperipheralpca' + title + '.txt',
                           zip(itercounts, master_errors_peripheralpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '0mastererrorstotalpca' + title + '.txt',
                           zip(itercounts, master_errors_totalpca))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '0mastererrorsspring' + title + '.txt',
                           zip(itercounts, master_errors_spring))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '0mastererrorstutte' + title + '.txt',
                           zip(itercounts, master_errors_tutte))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno

            try:
                np.savetxt(master_directory + '/' + '1peripheralpca_runtimes_' + title + '.txt',
                           zip(itercounts, peripheralpca_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '1totalpca_runtimes_' + title + '.txt',
                           zip(itercounts, totalpca_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '1spring_runtimes_' + title + '.txt',
                           zip(itercounts, spring_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            try:
                np.savetxt(master_directory + '/' + '1tutte_runtimes_' + title + '.txt',
                           zip(itercounts, tutte_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno

            try:
                np.savetxt(master_directory + '/' + '1face_enumeration_runtimes_' + title + '.txt',
                           zip(itercounts, face_enumeration_run_times))
            except:
                frameinfo = getframeinfo(currentframe())
                print frameinfo.filename, frameinfo.lineno
            i += 1
        except:
            print 'error with this attempt'
            pass
        # except:
        #     frameinfo = getframeinfo(currentframe())
        #     print frameinfo.filename, frameinfo.lineno
        #     print 'ERROR'
        #     # ipdb.set_trace()
        #     pass

        # timeout = 3
        # print 'cd to ' + master_directory + '? (Y/n)'
        # rlist, _, _ = select([sys.stdin], [], [], timeout)
        # if rlist:
        #     cd_query = sys.stdin.readline()
        #     # ipdb.set_trace()
        #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        #     if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
        #         os.chdir(master_directory)
        #     else:
        #         print 'no cd'
        # else:
        #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        #     os.chdir(master_directory)




def levenshtein_comparison_with_distortion():
    min = 250
    max = 260
    step = 10
    title = 'master_levenshtein_folder'
    repeats_per_step = 1

    reload(parsift_lib)
    master_directory = parsift_lib.prefix(title)
    number_of_steps = (max - min) / step
    polony_counts = np.zeros((number_of_steps * repeats_per_step))

    master_errors_spring = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_spring = np.zeros((number_of_steps * repeats_per_step,1))
    master_errors_tutte = np.zeros((number_of_steps * repeats_per_step))
    master_levenshtein_tutte = np.zeros((number_of_steps * repeats_per_step,1))


    for i in range(0, number_of_steps * repeats_per_step):
        try:
            npolony = min + i / repeats_per_step * step
            nsites  = (min + i / repeats_per_step * step)*24
            # substep = i%3
            print npolony
            polony_counts[i] = npolony
            # master_errors_spring[i], master_errors_tutte[i]
            basic_circle, spring_reconstruction, tutte_reconstruction, master_errors_spring[i], master_errors_tutte[
                i] = minimal_reconstruction_oop(target_nsites=nsites, npolony=npolony, randomseed=i % repeats_per_step,
                                                filename='monalisacolor.png', master_directory=master_directory,
                                                iterlabel=str(npolony))

            master_levenshtein_spring[i] = spring_reconstruction.levenshtein_distance
            master_levenshtein_tutte[i] = tutte_reconstruction.levenshtein_distance
            # ipdb.set_trace()
            np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
            np.savetxt(master_directory + '/' + 'master_levenshtein_spring' + title + '.txt', zip(polony_counts, master_levenshtein_spring))
            np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
            np.savetxt(master_directory + '/' + 'master_levenshtein_tutte' + title + '.txt', zip(polony_counts, master_levenshtein_tutte))
        except:pass

    timeout = 3
    print 'cd to ' + master_directory + '? (Y/n)'
    rlist, _, _ = select([sys.stdin], [], [], timeout)
    if rlist:
        cd_query = sys.stdin.readline()
        # ipdb.set_trace()
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        if cd_query == 'y\n' or cd_query == 'yes\n' or cd_query == 'Y\n' or cd_query == 'YES\n' or cd_query == 'Yes\n' or cd_query == '\n':
            os.chdir(master_directory)
        else:
            print 'no cd'
    else:
        print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
        os.chdir(master_directory)









# def minimal_reconstruction_oop(target_nsites = 300 ,npolony = 50,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
#     reload(parsift_lib)
#     # directory_name = master_directory + '/' + directory_label
#     if not master_directory:
#         directory_name = parsift_lib.prefix(directory_label)
#     else:
#         directory_name = master_directory
#     t0 = time.time()
#     basic_circle = parsift_lib.Surface(target_nsites = target_nsites, directory_label='',master_directory=directory_name)
#     basic_circle.circular_grid_ian()
#     basic_circle.scale_image_axes(filename = 'monalisacolor.png')
#     basic_circle.seed(nseed=npolony, full_output=False)
#     basic_circle.crosslink_polonies(full_output=False)
#     surface_prep_t = time.time() - t0
#
#
#     #ideal_graph = litesim.get_ideal_graph(delaunay)
#     # basic_circle.conduct_embeddings(full_output=False,image_only=False)
#
#     spring_reconstruction = parsift_lib.Reconstruction(basic_circle.directory_name,
#                                                        corseed=basic_circle.corseed,
#                                                        ideal_graph = basic_circle.ideal_graph,
#                                                        untethered_graph = basic_circle.untethered_graph,
#                                                        rgb_stamp_catalog = basic_circle.rgb_stamp_catalog)
#     springt0 = time.time()
#     spring_reconstruction.conduct_spring_embedding(full_output=False,image_only=False)
#     springtime = time.time() - springt0
#
#     spring_reconstruction.align(title='align_spring_points'+iterlabel, error_threshold=1, full_output=True)
#     spring_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(spring_reconstruction.reconstructed_pos)[0], basic_circle.untethered_graph)
#
#     spring_err = parsift_lib.get_radial_profile(spring_reconstruction.reconstructed_points, spring_reconstruction.distances, title='rad_spring'+iterlabel)
#     tuttet0 = time.time()
#     tutte_reconstruction = parsift_lib.Reconstruction(basic_circle.directory_name,
#                                                        corseed=basic_circle.corseed,
#                                                        ideal_graph = basic_circle.ideal_graph,
#                                                        untethered_graph = basic_circle.untethered_graph,
#                                                        rgb_stamp_catalog = basic_circle.rgb_stamp_catalog)
#
#     tutte_reconstruction.conduct_tutte_embedding(full_output=False, image_only=False)
#     # tutte_reconstruction.face_enumeration_runtime
#     tuttetime = time.time() - tuttet0
#
#     tutte_reconstruction.align(title='align_tutte_points' + iterlabel, error_threshold=1, full_output=False)
#     tutte_reconstruction.get_delaunay_comparison(parsift_lib.positiondict_to_list(tutte_reconstruction.reconstructed_pos)[0],
#                                                   basic_circle.untethered_graph)
#     tutte_err = parsift_lib.get_radial_profile(tutte_reconstruction.reconstructed_points, tutte_reconstruction.distances, title='rad_tutte'+iterlabel)
#
#     tutte_reconstruction.conduct_tutte_embedding_from_ideal(full_output=False,image_only=False)
#
#     return basic_circle, spring_reconstruction, tutte_reconstruction, spring_err, tutte_err, springtime, tuttetime, tutte_reconstruction.face_enumeration_runtime

















################################ <> <> <> <> <> <> <> <> <> <> <> <> ################################

#The basic starter template - use these lines to get a basic reconstruction dataset
#only construct image, embeddings,
#no alignment, no same-delaunay comparison

# reload(parsift_lib)
# #basic run
# target_nsites = 500
# npolony = 100
# basic_circle = parsift_lib.Surface(target_nsites = target_nsites, directory_label='oop_testing')
# basic_circle.circular_grid_ian()
# basic_circle.scale_image_axes(filename = 'monalisacolor.png')
# basic_circle.seed(nseed=npolony, full_output=False)
# basic_circle.crosslink_polonies(full_output=True)
# basic_circle.conduct_embeddings(full_output=True,image_only=False)

################################ <> <> <> <> <> <> <> <> <> <> <> <> ################################




# def vary_polony_no(xlen=100, ylen=100, min=10,max=50,step=10,title='polony_variation',repeats_per_step=3):
#     reload(litesim)
#     master_directory = litesim.prefix(title)
#     number_of_steps = (max-min)/step
#     master_errors_spring = np.zeros((number_of_steps*repeats_per_step))
#     master_errors_tutte = np.zeros((number_of_steps*repeats_per_step))
#     polony_counts = np.zeros((number_of_steps*repeats_per_step))
#     for i in range(0,number_of_steps*repeats_per_step):
#         npolony = min + i/repeats_per_step*step
#         # substep = i%3
#         print npolony
#         polony_counts[i] = npolony
#         try: master_errors_spring[i], master_errors_tutte[i] = minimal_reconstruction(xlen,ylen,ncell=1,npolony=npolony,randomseed=i%repeats_per_step,filename='monalisacolor.png',master_directory=master_directory,iterlabel=str(npolony))
#         except: pass
#         np.savetxt(master_directory + '/' + 'mastererrorsspring' + title + '.txt', zip(polony_counts, master_errors_spring))
#         np.savetxt(master_directory + '/' + 'mastererrorstutte' + title + '.txt', zip(polony_counts, master_errors_tutte))
#
#     timeout = 3
#     print 'cd to ' + master_directory + '? (Y/n)'
#     rlist, _, _ = select([sys.stdin], [], [], timeout)
#     if rlist:
#         cd_query = sys.stdin.readline()
#         # ipdb.set_trace()
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
#             os.chdir(master_directory)
#         else:print 'no cd'
#     else:
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         os.chdir(master_directory)

# def minimal_reconstruction(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
#     reload(litesim)
#     # directory_name = master_directory + '/' + directory_label
#     if not master_directory:
#         directory_name = litesim.prefix(directory_label)
#     else:
#         directory_name = master_directory
#     points = litesim.circular_grid_yunshi(xlen, ylen, randomseed=randomseed)
#     scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)
#     corseed, p1, bc, seed = litesim.seed(npolony, points,full_output=False)
#     p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=litesim.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis,full_output=False)
#     ideal_graph = litesim.get_ideal_graph(delaunay)
#     spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet, bc,
#                                                                                             group_cell=0, ideal_graph=ideal_graph,
#                                                                                             rgb_stamp_catalog=rgb_stamp_catalog,
#                                                                                           full_output=False)
#     spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
#     spring_distances = litesim.align(corseed, spring_points / np.max(np.sqrt(spring_points ** 2)),
#                                        reconstructed_graph=untethered_graph, title='align_spring_points'+iterlabel,
#                                        error_threshold=1, full_output=False)
#     spring_err =  litesim.get_radial_profile(spring_points, spring_distances, title='rad_spring'+iterlabel)
#
#     tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
#     tuttesamseq_distances = litesim.align(corseed, tuttesamseq_points, reconstructed_graph=untethered_graph,
#                                             title='align_tuttesamseq_points'+iterlabel, error_threshold=1.,
#                                             edge_number=len(tuttesamseq_points) * 3, full_output=False)
#     tutte_err = litesim.get_radial_profile(tuttesamseq_points, tuttesamseq_distances, title='rad_tutte'+iterlabel)
#     return spring_err, tutte_err
#
#     # timeout = 3
#     # print 'cd to ' + directory_name + '? (Y/n)'
#     # rlist, _, _ = select([sys.stdin], [], [], timeout)
#     # if rlist:
#     #     cd_query = sys.stdin.readline()
#     #     # ipdb.set_trace()
#     #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#     #     if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
#     #         os.chdir(directory_name)
#     #     else:print 'no cd'
#     # else:
#     #     print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#     #     os.chdir(directory_name)
#
# def image_only_reconstruct(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',master_directory = '', directory_label = '',randomseed=0,iterlabel=''):
#     reload(litesim)
#     # directory_name = master_directory + '/' + directory_label
#     if not master_directory:
#         directory_name = litesim.prefix(directory_label)
#     else:
#         directory_name = master_directory
#     points = litesim.circular_grid_yunshi(xlen, ylen, randomseed=randomseed)
#     scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)
#     corseed, p1, bc, seed = litesim.seed(npolony, points,full_output=False)
#     p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=litesim.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis,full_output=False)
#     ideal_graph = litesim.get_ideal_graph(delaunay)
#     spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet, bc,
#                                                                                             group_cell=0, ideal_graph=ideal_graph,
#                                                                                             rgb_stamp_catalog=rgb_stamp_catalog,
#                                                                                           full_output=False, image_only = True)
#     spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
#     spring_distances = litesim.align(corseed, spring_points / np.max(np.sqrt(spring_points ** 2)),
#                                        reconstructed_graph=untethered_graph, title='align_spring_points'+iterlabel,
#                                        error_threshold=1, full_output=False, image_only=True)
#     spring_err =  litesim.get_radial_profile(spring_points, spring_distances, title='rad_spring'+iterlabel)
#
#     tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
#     tuttesamseq_distances = litesim.align(corseed, tuttesamseq_points, reconstructed_graph=untethered_graph,
#                                             title='align_tuttesamseq_points'+iterlabel, error_threshold=1.,
#                                             edge_number=len(tuttesamseq_points) * 3, full_output=False,image_only=True)
#     tutte_err = litesim.get_radial_profile(tuttesamseq_points, tuttesamseq_distances, title='rad_tutte'+iterlabel)
#     return spring_err, tutte_err
#
# def full_experiment(xlen,ylen,ncell,npolony,filename = 'monalisacolor.png',directory_label = '',niter=1,randomseed=4):
#     reload(litesim)
#     directory_name = litesim.prefix(directory_label)
#     points=litesim.circular_grid_yunshi(xlen,ylen,randomseed=randomseed)
#     scaled_imarray, lookup_x_axis, lookup_y_axis = litesim.scale_image_axes(points, filename = filename)
#
#     corseed,p1,bc,seed=litesim.seed(npolony,points)
#     p2,p2_sec,p2_prim,cheating_sheet,delaunay,rgb_stamp_catalog=simulator.rca(p1,points,corseed,scaled_imarray, lookup_x_axis, lookup_y_axis)
#     ideal_graph = litesim.get_ideal_graph(delaunay)
#     hybrid=litesim.perfect_hybrid(p2,p2_sec,bc)
#     cell_points=litesim.cell(points,ncell,corseed,hybrid,xlen,ylen)
#     spring_pos, tuttesamseq_pos, ideal_pos, untethered_graph = litesim.cheating_alignment(cheating_sheet,bc,cell_points,ideal_graph,rgb_stamp_catalog,points,npolony)
#     # flat_tutte_pos, dictlistx, dictlisty = simulator.positiondict_to_list(tuttesamseq_pos)
#     # good_edges, bad_new_edges, missed_old_edges, cost = simulator.get_delaunay_comparison(simulator.positiondict_to_list(spring_pos)[0], untethered_graph)
#     # print cost
#     annealed_pos_spring = litesim.anneal_to_initial_delaunay(pos=spring_pos, untethered_graph = untethered_graph,niter=niter,name='spring')
#     annealed_pos_tutte = litesim.anneal_to_initial_delaunay(pos=tuttesamseq_pos, untethered_graph=untethered_graph, niter=niter,name='tutte')
#     annealed_pos_tutte_ideal = litesim.anneal_to_initial_delaunay(pos=ideal_pos, untethered_graph=ideal_graph,
#                                                               niter=niter, name='tutteideal')
#
#     ### get alignments with original positions ###
#     edge_number = len(cheating_sheet)
#     ideal_points = np.array(litesim.convert_dict_to_list(ideal_pos))
#     ideal_distances = litesim.align(corseed,ideal_points,title='align_ideal_points',error_threshold = 1.,edge_number = len(ideal_points)*3)
#     litesim.get_radial_profile(ideal_points,ideal_distances,title='radial_error_plot_ideal')
#
#     spring_points = np.array(litesim.convert_dict_to_list(spring_pos))
#     spring_distances = litesim.align(corseed,spring_points/np.max(np.sqrt(spring_points**2)),reconstructed_graph = untethered_graph,title='align_spring_points',error_threshold = 1)
#     litesim.get_radial_profile(spring_points, spring_distances, title='radial_error_plot_spring')
#
#     tuttesamseq_points = np.array(litesim.convert_dict_to_list(tuttesamseq_pos))
#     tuttesamseq_distances = litesim.align(corseed,tuttesamseq_points,reconstructed_graph = untethered_graph, title='align_tuttesamseq_points',error_threshold = 1.,edge_number = len(ideal_points)*3)
#     litesim.get_radial_profile(tuttesamseq_points,tuttesamseq_distances,title='radial_error_plot_tuttesamseq')
#
#     annealedsamseq_points_spring = np.array(litesim.convert_dict_to_list(annealed_pos_spring))
#     annealedsamseq_distances_spring = litesim.align(corseed,annealedsamseq_points_spring/np.max(np.sqrt(annealedsamseq_points_spring**2)),reconstructed_graph = untethered_graph, title='align_annealedsamseq_points_spring',error_threshold = 1.,edge_number = len(ideal_points)*3)
#     litesim.get_radial_profile(annealedsamseq_points_spring,annealedsamseq_distances_spring,title='radial_error_plot_annealedsamseq')
#     # ipdb.set_trace()
#     annealedsamseq_points_tutte = np.array(litesim.convert_dict_to_list(annealed_pos_tutte))
#     annealedsamseq_distances_tutte = litesim.align(corseed,annealedsamseq_points_tutte/np.max(np.sqrt(annealedsamseq_points_tutte**2)),reconstructed_graph = untethered_graph, title='align_annealedsamseq_points_tutte',error_threshold = 1.,edge_number = len(ideal_points)*3)
#     litesim.get_radial_profile(annealedsamseq_points_tutte,annealedsamseq_distances_tutte,title='radial_error_plot_annealedsamseq')
#
#     annealedsamseq_points_tutte_ideal = np.array(litesim.convert_dict_to_list(annealed_pos_tutte_ideal))
#     annealedsamseq_distances_tutte_ideal = litesim.align(corseed, annealedsamseq_points_tutte_ideal / np.max(
#         np.sqrt(annealedsamseq_points_tutte_ideal ** 2)), reconstructed_graph=ideal_graph,
#                                                      title='align_annealedsamseq_points_tutte_ideal', error_threshold=1.,
#                                                      edge_number=len(ideal_points) * 3)
#     litesim.get_radial_profile(annealedsamseq_points_tutte_ideal, annealedsamseq_distances_tutte_ideal,
#                                  title='radial_error_plot_annealedsamseq_ideal')
#
#
#     #change to the file directory at the end
#     timeout = 10
#     print 'cd to ' + directory_name + '? (Y/n)'
#     rlist, _, _ = select([sys.stdin], [], [], timeout)
#     if rlist:
#         cd_query = sys.stdin.readline()
#         # ipdb.set_trace()
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         if cd_query == 'y\n' or cd_query =='yes\n' or cd_query =='Y\n' or cd_query =='YES\n'  or cd_query =='Yes\n' or cd_query =='\n':
#             os.chdir(directory_name)
#         else:print 'no cd'
#     else:
#         print 'changing directories... use: "os.system("xdg-open nameoffile.png")" to view a file within shell'
#         os.chdir(directory_name)
#     # ipdb.set_trace()
#
