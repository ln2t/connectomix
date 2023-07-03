#!/usr/bin/env python3

# IMPORTS

from datetime import datetime
import os  # to interact with dirs (join paths etc)
import subprocess  # to call bin outside of python
from collections import defaultdict
import sys

# CLASSES


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    ERROR = BOLD + RED
    INFO = BOLD + GREEN
    WARNING = BOLD + YELLOW


class Report:
    """
    A class to create, update, modify and save reports with pretty stuff
    """
    def __init__(self, path=None, string=""):
        """
        path is the place where the report is written/updated
        """
        self.path = path
        self.string = string

    def init(self, subject, date_and_time, version,
             cmd, session=None):
        """
        Init the report with html headers and various information on the report
        """
        with open(self.path, "w") as f:
            f.write('<!DOCTYPE html>\n')
            f.write('<html>\n')
            f.write('<body>\n')
            f.write('<h1>Connectomix: individual report</h1>\n')
            f.write('<h2>Summary</h2>\n')
            f.write('<div>\n')
            f.write('<ul>\n')
            f.write(
                '<li>BIDS participant label: sub-%s</li>\n' % subject)
            if not session is None:
                f.write(
                    '<li>Session: ses-%s</li>\n' % session)
            f.write('<li>Date and time: %s</li>\n' % date_and_time)
            f.write('<li>Connectomix version: %s</li>\n' % version)
            f.write('<li>Command line options:\n')
            f.write('<pre>\n')
            f.write('<code>\n')
            f.write(str(cmd) + '\n')
            f.write('</code>\n')
            f.write('</pre>\n')
            f.write('</li>\n')
            f.write('</ul>\n')
            f.write('</div>\n')

    def add_section(self, title):
        """
        Add a section to the report
        """
        with open(self.path, "a") as f:
            f.write('<h2>%s</h2>\n' % title)

    def add_subsection(self, title):
        """
        Add a subsection to the report
        """
        with open(self.path, "a") as f:
            f.write('<h3>%s</h3>\n' % title)

    def add_sentence(self, sentence):
        """
        Add a sentence to the report
        """
        with open(self.path, "a") as f:
            f.write('%s<br>\n' % sentence)

    def add_png(self, path):
        """
        Add an image to the report from a path to the png file
        """
        with open(self.path, "a") as f:
            f.write('<img width="400" src = "%s">' % path)

    def append(self, string):
        """
        Append string to report
        """
        with open(self.path, "a") as f:
            f.write(string)


    def finish(self):
        """
        Writes the last lines of the report to finish it.
        """
        with open(self.path, "a") as f:
            f.write('</body>\n')
            f.write('</html>\n')

# FUNCTIONS


def msg_info(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.INFO}INFO{BColors.ENDC} " + time_stamp + msg, flush=True)


def msg_error(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.ERROR}ERROR{BColors.ENDC} " + time_stamp + msg, flush=True)


def msg_warning(msg):
    now = datetime.now()
    time_stamp = now.strftime("(%Y-%m-%d-%H-%M-%S) ")
    print(f"{BColors.WARNING}WARNING{BColors.ENDC} " + time_stamp + msg, flush=True)


def printProgressBar (iteration, total, prefix = '', suffix = 'Complete', decimals = 4, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    complete_prefix = f"{BColors.OKCYAN}Progress {BColors.ENDC}" + prefix
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{complete_prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush="True")
    # Print New Line on Complete
    if iteration == total:
        print()


def run(command, env={}):
    """Execute command as in a terminal

    Also prints any output of the command into the python shell
    Inputs:
        command: string
            the command (including arguments and options) to be executed
        env: to add stuff in the environment before running the command

    Returns:
        nothing
    """

    # Update env
    merged_env = os.environ
    merged_env.update(env)

    # Run command
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)

    # Read whatever is printed by the command and print it into the
    # python console
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d" % process.returncode)


def nested_dict(n, type):
    """

    :param n: an integer defining the nesting level
    :param type: type of nested object, typically list or float
    :return: a nested dictionary
    """
    if n == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: nested_dict(n-1, type))


def get_graphics(data, title, labels, coords):
    """

    :param data: numpy array with data
    :param title: title for the graphics
    :param labels: labels for the carpet plot
    :param coords: coordinates for the connectome graphs
    :return: dict with three fields:
                carpet: a matplotlib axes object
                connectome: a matplotlib figure object
                view: a nilearn view object
    """
    from nilearn import plotting
    from matplotlib import pyplot as plt
    matrix = plotting.plot_matrix(data, figure=(10, 8), labels=labels,
                                  vmax=0.8, vmin=-0.8, title=title,
                                  reorder=False)  # save with carpet.figure.savefig(path)
    connectome = plotting.plot_connectome(data, coords, title=title)  # save with connectome.savefig(path)
    #view = plotting.view_connectome(data, coords,
    #                                title=title)  # put in html by inserting the string view._repr_html_()

    plt.close()

    return dict(matrix=matrix.figure, connectome=connectome)

def get_masker_labels_coords(seeds):

    from nilearn import datasets
    from nilearn import plotting
    from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker

    if seeds['type'] == 'msdl_atlas':
        # we load a probabilistic atlas
        dataset = datasets.fetch_atlas_msdl()
        masker = NiftiMapsMasker(maps_img=dataset.maps, standardize=True)
        labels = dataset.labels
        coordinates = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

        # for non-probabilistic atlases, use:
        # dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
        # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
        # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    if seeds['type'] == 'customseeds':
        masker = NiftiSpheresMasker(
            seeds['coordinates'], radius=seeds['radius'], detrend=True, standardize=True,
            low_pass=0.1, high_pass=0.01, t_r=2,
            memory='nilearn_cache', memory_level=1, verbose=2)
        labels = seeds['labels']
        coordinates = seeds['coordinates']

    return masker, labels, coordinates

def get_timeseries(layout, filter, strategy, seeds):
    """
    Compute connectomes and graphs of nodes for various denoising strategies. Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param layout: BIDS layout containing the fmriprep outputs
    :param filter: dict to be used as a BIDS filter to select participants data
    :param strategy: str setting the denoising strategy
    :param seeds: dict to set regions to use as seeds. If seeds['type'] = 'msdl_atlas', will use probabilistic mdsl atlas. Other wise: will read labels, coordinates and radius fields.
    :return: dict with three fields: correlation, covariance and precision
    """

    # imports

    from nilearn.interfaces.fmriprep import load_confounds_strategy

    filter.update({'suffix': "bold"})
    if strategy == 'ica_aroma':
        filter.update({'desc': "smoothAROMAnonaggr"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy}
    else:
        filter.update({'desc': "preproc"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}
    try:
        fmri_files = layout.get(**filter, extension='.nii.gz')[0]
    except IndexError:
        msg_error('fmri file not found!')
        sys.exit(1)
    confounds, sample_mask = load_confounds_strategy(fmri_files, **options_for_load_confounds_strategy)

    masker, labels, coords = get_masker_labels_coords(seeds)

    try:
        time_series = masker.fit_transform(fmri_files, confounds=confounds)
    except:
        msg_error('There is a problem with series %s. Check QC of these data!' % fmri_files)
        sys.exit(1)

    return time_series, labels, coords

def get_connectome(layout, filter, strategy, seeds):
    """
    Compute connectomes and graphs of nodes for various denoising strategies. Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param layout: BIDS layout containing the fmriprep outputs
    :param filter: dict to be used as a BIDS filter to select participants data
    :param strategy: str setting the denoising strategy
    :param seeds: dict to set regions to use as seeds. If seeds['type'] = 'msdl_atlas', will use probabilistic mdsl atlas. Other wise: will read labels, coordinates and radius fields.
    :return: dict with three fields: correlation, covariance and precision
    """

    # imports
    global correlation_matrix, caption, labels, coordinates, masker, fmri_files
    from nilearn import datasets
    from nilearn.interfaces.fmriprep import load_confounds_strategy
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np
    from nilearn import plotting
    from nilearn.maskers import NiftiMapsMasker, NiftiSpheresMasker

    filter.update({'suffix': "bold"})
    if strategy == 'ica_aroma':
        filter.update({'desc': "smoothAROMAnonaggr"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy}
    else:
        filter.update({'desc': "preproc"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}
    try:
        fmri_files = layout.get(**filter, extension='.nii.gz')[0]
    except:
        msg_error('fmri file %s not found' % fmri_files)
        sys.exit(1)
    confounds, sample_mask = load_confounds_strategy(fmri_files, **options_for_load_confounds_strategy)

    if seeds['type'] == 'msdl_atlas':

        # we load a probabilistic atlas
        dataset = datasets.fetch_atlas_msdl()
        masker = NiftiMapsMasker(maps_img=dataset.maps, standardize=True)
        labels = dataset.labels
        coordinates = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

        # for non-probabilistic atlases, use:
        # dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
        # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
        # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)

    if seeds['type'] == 'customseeds':
        masker = NiftiSpheresMasker(
            seeds['coordinates'], radius=seeds['radius'], detrend=True, standardize=True,
            low_pass=0.1, high_pass=0.01, t_r=2,
            memory='nilearn_cache', memory_level=1, verbose=2)
        labels = seeds['labels']
        coordinates = seeds['coordinates']

    try:
        time_series = masker.fit_transform(fmri_files, confounds=confounds)
    except:
        msg_error('There is a problem with series %s. Check QC of these data!' % fmri_files)
        sys.exit(1)

    correlation_matrix = ConnectivityMeasure(kind='correlation').fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 0)  # for visual purposes
    correlation_graphics = get_graphics(correlation_matrix, 'correlation' + ' - ' + strategy, labels, coordinates)

    return dict(type=seeds['type'], data=correlation_matrix, graphics=correlation_graphics)

def get_connectome_bk(layout, filter, strategy, nodes_coords, sphere_radius, nodes_labels):
    """
    Compute connectomes and graphs of nodes for various denoising strategies. Results (mostly: figures) are saved as png in output_dir, and stuff is also added in the report.

    :param layout: BIDS layout containing the fmriprep outputs
    :param filter: dict() to be used as a BIDS filter to select participants data
    :param output_dir: directory where images will be saved
    :param report: report in which various figures and results will be saved. Must be initialized beforehand (report.init()).
    :param nodes: array of 3D coordinates for node
    :param radius: radius of the spheres centered on the nodes
    :return: dict with three fields: correlation, covariance and precision
    """

    # imports
    from nilearn import datasets
    from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
    from nilearn.connectome import ConnectivityMeasure
    import numpy as np
    from nilearn import plotting
    from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker, NiftiSpheresMasker
    from sklearn.covariance import GraphicalLassoCV

    # atlas fetching

    # example of non-probabilistic atlases:
    #dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

    # we load a probabilistic atlas
    dataset = datasets.fetch_atlas_msdl()
    atlas_filename = dataset.maps
    labels = dataset.labels

    # for non-probabilistic atlases, use:
    # atlas_region_coords = plotting.find_parcellation_cut_coords(dataset.maps)
    # atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)
    atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(dataset.maps)

    nodes_labels = dataset.labels + nodes_labels
    nodes_coords = np.concatenate((plotting.find_probabilistic_atlas_cut_coords(dataset.maps), np.array(nodes_coords)))

    correlation_measure = ConnectivityMeasure(kind='correlation')

    filter.update({'suffix': "bold"})

    if strategy == 'ica_aroma':
        # filter.update({'space': 'MNI152NLin6Asym'})
        filter.update({'desc': "smoothAROMAnonaggr"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy}
    else:
        filter.update({'desc': "preproc"})
        options_for_load_confounds_strategy = {'denoise_strategy': strategy, 'motion': 'basic'}

    # get the files
    fmri_files = layout.get(**filter, extension='.nii.gz')[0]

    # for non-probabilistic atlases, use:
    # masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
    masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True)
    masker.fit(fmri_files)

    # we use pre-defined strategies
    confounds, sample_mask = load_confounds_strategy(fmri_files, **options_for_load_confounds_strategy)

    # extract now the time series with the current denoising strategy
    time_series = masker.fit_transform(fmri_files,
                                       confounds=confounds,
                                       sample_mask=sample_mask)

    # now that we have the time series, we can compute various stuff

    # correlation
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    # for visual purposes we fill the diagonal with zeros
    np.fill_diagonal(correlation_matrix, 0)

    # covariance and precision
    # todo: understand the technical difference between the correlation matrix and the covariance matrix
    estimator = GraphicalLassoCV()
    estimator.fit(time_series)

    covariance_matrix = estimator.covariance_
    np.fill_diagonal(covariance_matrix, 0)

    precision_matrix = -estimator.precision_
    np.fill_diagonal(precision_matrix, 0)

    # for each of these measurements, we make two kinds of plots: a carpet-like plot and connectome plots (showing the nodes with their connection in the brain).
    # moreover, for the connectome, we can also make nice 3D dynamic figures (called a 'view' in nilearn).

    correlation = get_graphics(correlation_matrix, 'correlation' + ' - ' + strategy, labels, atlas_region_coords)
    covariance = get_graphics(covariance_matrix, 'covariance' + ' - ' + strategy, labels, atlas_region_coords)
    precision = get_graphics(precision_matrix, 'precision' + ' - ' + strategy, labels, atlas_region_coords)

    # now we compute stuff fot the list of nodes with ROI = spheres of radius sphere_radius

    masker_nodes = NiftiSpheresMasker(
        nodes_coords, radius=sphere_radius, detrend=True, standardize=True,
        low_pass=0.1, high_pass=0.01, t_r=2,
        memory='nilearn_cache', memory_level=1, verbose=2)

    nodes_time_series = masker_nodes.fit_transform(fmri_files, confounds=[confounds])
    nodes_connectivity_measure = ConnectivityMeasure(kind='partial correlation')
    nodes_partial_correlation_matrix = nodes_connectivity_measure.fit_transform(
        [nodes_time_series])[0]

    nodes = get_graphics(nodes_partial_correlation_matrix, 'partial correlation' + ' - ' + strategy, nodes_labels, nodes_coords)

    from nilearn import plotting
    nodes_carpet = plotting.plot_matrix(nodes_partial_correlation_matrix, figure=(10, 8), labels=nodes_labels,
                                  vmax=0.8, vmin=-0.8,
                                  reorder=False)  # save with carpet.figure.savefig(path)

    return dict(correlation=correlation, covariance=covariance, precision=precision, nodes=nodes, nodes_plot=nodes_carpet, nodes_matrix=nodes_partial_correlation_matrix)
