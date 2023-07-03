#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""

# imports
import argparse  # to deal with all the app arguments
from os.path import join  # to build input/tmp/output paths
from bids import BIDSLayout as bidslayout  # to handle BIDS data
from utils.utils import *  # custom utilities from the package
from pathlib import Path  # to create dirs
import sys  # to exit
import csv
import numpy as np
from nilearn.connectome import ConnectivityMeasure

space = 'MNI152NLin6Asym'
# space = 'MNI152NLin2009cAsym'

no_session_flag = 4.2  # used to for dummy value of session variable

def main():

    __version__ = open(join(os.path.dirname(os.path.realpath(__file__)),
                            'version')).read()
    msg_info("Version: %s"%__version__)

    #  setup the arguments
    parser = argparse.ArgumentParser(description = 'Entrypoint script.')
    parser.add_argument('bids_dir', help='The directory with the input '
                                         'dataset formatted according to '
                                         'the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output '
                                           'files will be stored.')
    parser.add_argument('analysis_level', help='Level of the analysis that '
                                               'will be performed. '
                                               'Multiple participant level '
                                               'analyses can be run '
                                               'independently (in parallel)'
                                               ' using the same '
                                               'output_dir.',
                        choices = ['participant',
                                   'group'])
    parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label corresponds to sub-<participant_label> from the BIDS spec (so it does not include "sub-"). If this parameter is not provided all subjects should be analyzed. Multiple participants can be specified with a space separated list.', nargs = "+")
    parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation', action='store_true')
    parser.add_argument('--fmriprep_dir', help='Path of the fmriprep derivatives. If ommited, set to bids_dir/derivatives/fmriprep')
    parser.add_argument('--task', help='Name of the task to be used. If omitted, will search for \'restingstate\'.')
    parser.add_argument('--sessions', help='Name of the session to consider. If omitted, will loop over all sessions.', nargs = "+")
    parser.add_argument('--space', help='Name of the space to be used. Must be associated with fmriprep output.')
    parser.add_argument('--denoising_strategies', help='Names of the denoising strategies to consider. If ommited, set to \'simple\'. Examples are \'simple\', \'scrubbing\', \'compcor\', \'ica_aroma\'', nargs="+")
    parser.add_argument('--seeds_file',
                        help='Optional. Path to a .tsv file from which the nodes to compute the connectome will be loaded. The .tsv file should have four columns, without header. The first one contains the node name (a string) and the three others are the x,y,z coordinates in MNI space of the correspondings node. If omitted, it will load the msdl probabilitic atlas (see nilearn documentations for more info). ')
    parser.add_argument('-v', '--version', action='version', version='BIDS-App example version {}'.format(__version__))
    #  parse
    args = parser.parse_args()

    # check 1: bids_dir exists?
    if not os.path.isdir(args.bids_dir):
        msg_error("Bids directory %s not found." % args.bids_dir)
        sys.exit(1)
    else:
        #  fall back to default if argument not provided
        bids_dir = args.bids_dir

    #if not args.skip_bids_validator:
        # todo: check if bids-validator is in path. Alternative: directly
        #  call the appropriate pip package - see example in fmriprep
        # todo: restore this part.
        # run('bids-validator --ignoreWarnings %s' % args.bids_dir)
    #else:
    #    msg_info("Skipping bids-validation")

    # initiate BIDS layout
    msg_info("Indexing BIDS dataset...")
    layout = bidslayout(args.bids_dir)

    # check : valid subjects?
    subjects_to_analyze = []

    if args.participant_label:  # only for a subset of subjects
        subjects_to_analyze = args.participant_label
        # in that case we must ensure the subjects exist in the layout:
        for subj in subjects_to_analyze:
            if subj not in layout.get_subjects():
                msg_error("Subject %s in not included in the "
                          "bids database." % subj)
                sys.exit(1)
    else:  # for all subjects
        subjects_to_analyze = layout.get_subjects()

    #  fmriprep dir definition
    if args.fmriprep_dir:
        fmriprep_dir = args.fmriprep_dir
    else:
        fmriprep_dir = join(args.bids_dir, "derivatives", "fmriprep")

    # check: fmriprep dir exists?
    if not os.path.isdir(fmriprep_dir):
        msg_error("fmriprep dir %s not found." % fmriprep_dir)
        sys.exit(1)

    seeds = dict()
    if args.seeds_file:
        if not os.path.isfile(args.seeds_file):
            msg_error('Node file %s not found.' % args.seeds_file)
            sys.exit(1)
        seeds['type'] = 'customseeds'
        seeds['radius'] = 5
        with open(args.seeds_file) as file:
            tsv_file = csv.reader(file, delimiter="\t")
            seeds['labels'] = []
            seeds['coordinates'] = []
            for line in tsv_file:
                seeds['labels'].append(line[0])
                seeds['coordinates'].append(np.array(line[1:4], dtype=int))
            #todo: add checks that what we are reading make sense
    else:
        seeds['type'] = 'msdl_atlas'

    layout.add_derivatives(fmriprep_dir)

    # check: valid task?
    if args.task:
        if args.task not in layout.get_tasks():
            msg_error("Selected task %s is not in the BIDS dataset. "
                      "Available tasks are %s." % (args.task,
                                                   layout.get_tasks()))
            sys.exit(1)
        task = args.task
    else:
        # fall back to default value
        task = "restingstate"

    # check: valid sessions?
    if args.sessions:
        for ses in args.sessions:
            if ses not in layout.get_sessions():
                msg_error("Selected session %s is not in the BIDS dataset. "
                          "Available session are %s." % (ses,
                                                       layout.get_sessions()))
                sys.exit(1)
        sessions = args.sessions

    available_strategies = ['simple', 'scrubbing', 'compcor', 'ica_aroma']  # those are the four standard strategies directly available in nilearn

    if args.denoising_strategies:
        for strategy in args.denoising_strategies:
            if strategy not in available_strategies:
                msg_error('Selected strategy %s is not available. Available strategies are %s.' % (strategy, available_strategies))
                sys.exit(1)
        strategies = args.denoising_strategies
    else:
        strategies = ['simple']

    msg_info('Denoising strategie(s) are set to %s' % strategies)

    # create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # initiate dataset_description file for outputs
    dataset_description = os.path.join(args.output_dir,
                                       'dataset_description.json')
    with open(dataset_description, 'w') as ds_desc:
        # todo: get the correct BIDS version
        ds_desc.write(('{"Name": "connectomix", "BIDSVersion": "x.x.x", '
                       '"DatasetType": "derivative", "GeneratedBy": '
                       '[{"Name": "connectomix"}, {"Version": "%s"}]}')
                      % __version__.rstrip('\n'))
        ds_desc.close()

    # add output dir as BIDS derivatives in layout
    layout.add_derivatives(args.output_dir)

    # get absolute path for our outputs
    output_dir = layout.derivatives['connectomix'].root

    # print some summary before running
    msg_info("Bids directory: %s" % bids_dir)
    msg_info("Fmriprep directory: %s" % fmriprep_dir)
    msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    msg_info("Task to analyse: %s" % task)

    # nodes coordinates and labels for seed-based connectivity

    dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
    dmn_labels = [
        'Posterior Cingulate Cortex',
        'Left Temporoparietal junction',
        'Right Temporoparietal junction',
        'Medial prefrontal cortex',
    ]

    # cerebellum seeds
    #
    # cerebellum_coords = [(-15, -61, -36), (15, -61, -36),
    #                      (-36, -74, -46), (42, -72, -48),
    #                      (-34, -66, -32), (28, -68, -30),
    #                      (-6, -86, -32), (6, -82, -28),
    #                      (-14, -48, -14), (14, -46, -14),
    #                      (-20, -60, -16), (18, -58, -14),
    #                      (-22, -52, -60), (24, -62, -58),
    #                      ]
    # cerebellum_labels = [
    #     'Left Dentate Nuclei',
    #     'Right Dentate Nuclei',
    #     'Left Crus I',
    #     'Right Crus I',
    #     'Left Crus II',
    #     'Right Crus II',
    #     'Left Lobule VIIa',
    #     'Right Lobule VIIa',
    #     'Left Lobule V',
    #     'Right Lobule V',
    #     'Left Lobule VI',
    #     'Right Lobule VI',
    #     'Left Lobule VIII',
    #     'Right Lobule VIII',
    # ]

    # main section

    # running participant level
    if args.analysis_level == "participant":

        # loop over subject to analyze
        for subject_label in subjects_to_analyze:

            # find all sessions
            if len(layout.get_sessions(subject=subject_label)) == 0:
                sessions = [no_session_flag]
            else:
                if not args.sessions:
                    sessions = layout.get_sessions(subject=subject_label)
                msg_info('Found the following sessions for this subject: %s' % sessions)

            for session in sessions:

                # init a basic filter for layout.get()
                filter = {}

                # add participant
                filter.update(subject=subject_label)

                # add session, if any
                if not session == no_session_flag:
                    filter.update(session=session)
                    msg_info('Running for session ses-%s' % session)

                # return_type will always be set to filename
                filter.update(return_type='filename')

                # add space to filter
                filter.update(space=space)

                msg_info("Running for participant %s" % subject_label)

                # create output_dir/sub-XXX directory

                if session == no_session_flag:
                    subject_output_dir = os.path.join(output_dir,
                                                  "sub-" + subject_label)
                else:
                    subject_output_dir = os.path.join(output_dir,
                                                      "sub-" + subject_label, "ses-" + session)

                Path(subject_output_dir).mkdir(parents=True, exist_ok=True)

                # subject_figure_dir = os.path.join(subject_output_dir,
                #                                   'figures')
                # Path(subject_figure_dir).mkdir(parents=True, exist_ok=True)

                # # set paths for various outputs filenames:
                # outputs_fn = {}

                # if session == no_session_flag:
                #     subject_prefix = os.path.join(subject_output_dir,
                #                       "sub-" + subject_label)
                # else:
                #     subject_prefix = os.path.join(subject_output_dir,
                #                                   "sub-" + subject_label + "_ses-" + session)

                # prefix = subject_prefix + "_space-" + space + '_'
                # report_extension = '.html'
                # outputs['report'] = prefix + 'report' + report_extension

                #outputs['nodes_matrix'] = prefix + 'nodes_data.txt'

                # add subject to filter
                filter.update(subject=subject_label)

                # add task to filter
                filter.update(task=task)

                # initiate report
                # report = Report(
                #     outputs['report'])

                # if session == no_session_flag:
                #     report.init(subject=subject_label,
                #             date_and_time=datetime.now(),
                #             version=__version__, cmd=args.__dict__)
                # else:
                #     report.init(subject=subject_label,
                #                 session=session,
                #                 date_and_time=datetime.now(),
                #                 version=__version__, cmd=args.__dict__)

                # todo: check that the data are present (in particular the ica-aroma denoised stuff)

                results = dict()
                time_series = dict()
                # quantities_of_interest = ['correlation', 'covariance', 'precision']
                # quantities_of_interest = ['correlation']

                for strategy in strategies:

                    msg_info('Starting computations for strategy %s' % strategy)
                    results[strategy] = dict()

                    # results[strategy] = get_connectome(layout, filter, strategy, seeds)

                    results[strategy]['timeseries'], labels, coords = get_timeseries(layout, filter, strategy, seeds)

                    correlation_matrix = ConnectivityMeasure(kind='correlation').fit_transform([results[strategy]['timeseries']])[0]
                    np.fill_diagonal(correlation_matrix, 0)  # for visual purposes

                    results[strategy]['data'] = correlation_matrix
                    results[strategy]['graphics'] = get_graphics(correlation_matrix, 'correlation' + ' - ' + strategy, labels, coords)

                    # define output paths
                    prefix = os.path.join(subject_output_dir, "sub-" + subject_label)
                    # add session, if necessary
                    if not session == no_session_flag:
                        prefix = prefix + "_ses-" + session

                    # add description
                    prefix = prefix + "_desc-" + seeds['type']

                    outputs_fn = {}

                    # save a couple of nice images
                    for item in ['matrix', 'connectome']:
                        outputs_fn[item] = prefix + '_' + item + '.png'
                        results[strategy]['graphics'][item].savefig(outputs_fn[item])

                    # save the matrix data and timeseries
                    for item in ['data', 'timeseries']:
                        outputs_fn[item] = prefix + '_' + item + '.tsv'
                        np.savetxt(outputs_fn[item], results[strategy][item], delimiter='\t')

                # now we build the report. We want one col for each strategy

                # strategy_html_headers = '<span style="padding-left:150px">simple</span><span style="padding-left:300px">scrubbing</span><span style="padding-left:300px">compcor</span><span style="padding-left:300px">ica-aroma</span>'
                # report.add_section(title='Carpet plots')
                # for item in quantities_of_interest:
                #     report.add_subsection(title=item)
                #     # report.add_sentence(sentence=strategy_html_headers)
                #     for strategy in strategies:
                #         report.add_png(outputs[(strategy, item, 'carpet')])
                #
                # report.add_section(title='Connectome plots')
                # for item in quantities_of_interest:
                #     report.add_subsection(title=item)
                #     # report.add_sentence(sentence=strategy_html_headers)
                #     for strategy in strategies:
                #         report.add_png(outputs[(strategy, item, 'connectome')])
                #
                # if include_3D_plot_flag:
                #     report.add_section(title='Connectome 3D views')
                #     for item in quantities_of_interest:
                #         report.add_subsection(title=item)
                #         # report.add_sentence(sentence=strategy_html_headers)
                #         for strategy in strategies:
                #             report.append(results[strategy][item]['view']._repr_html_())
                #
                # report.add_section(title='Carpet plots for provided nodes')
                # for strategy in strategies:
                #     report.add_png(os.path.join(subject_figure_dir, strategy + '_' + 'nodes_plot.png'))
                #
                # report.add_section(title='Connectome plots for provided nodes')
                # for strategy in strategies:
                #     report.add_png(outputs[('nodes_connectome', strategy)])
                #
                # report.finish()

    # running group level
    elif args.analysis_level == "group":

        time_series = dict()
        filter = dict(suffix='timeseries', extension='.tsv', return_type='filename')
        msg_warning('Datasets without session not supported yet.')

        group_output_dir = os.path.join(output_dir, 'group')
        Path(group_output_dir).mkdir(parents=True, exist_ok=True)

        for session in layout.derivatives['connectomix'].get_sessions():
            time_series[session] = dict()
            msg_warning('This tool should not be used if you have run several denoising strategies and the outputs are coexisting in the derivative folder.')
            for subject in layout.derivatives['connectomix'].get_subjects(session=session):
                try:
                    series_fn = layout.derivatives['connectomix'].get(**filter, subject=subject, session=session, desc=seeds['type'])[0]
                except:
                    msg_warning('No series found for subject %s, skipping.' % subject)
                    continue
                time_series[session][subject] = np.genfromtxt(series_fn, delimiter='\t')

            time_series_stack = []

            for key in time_series[session].keys():
                time_series_stack.append(time_series[session][key])


            measure = ConnectivityMeasure(kind='tangent')
            connectivities = measure.fit(time_series_stack)
            group_connectivity = measure.mean_
            data = measure.fit_transform(time_series_stack)[0]
            np.fill_diagonal(data, 0)  # for visual purposes
            graphics = get_graphics(data, 'Group connectome, ses-%s' % session, seeds['labels'], seeds['coordinates'])
            graphics['matrix']

            group_prefix = os.path.join(group_output_dir, 'ses-%s' % session)

            outputs_fn = dict()
            # save the graphics
            for item in ['matrix', 'connectome']:
                outputs_fn[item] = group_prefix + '_' + item + '.png'
                graphics[item].savefig(outputs_fn[item])

            # save the matrix data and timeseries
            outputs_fn['data'] = group_prefix + '_data.tsv'
            np.savetxt(outputs_fn['data'], data, delimiter='\t')

    msg_info("The End!")

if __name__ == '__main__':
    main()
