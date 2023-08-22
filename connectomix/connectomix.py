#!/usr/bin/env python3
"""BIDS app to compute connectomes from fmri data preprocessed with FMRIPrep

Author: Antonin Rovai

Created: August 2022
"""

# imports
from bids import BIDSLayout as bidslayout  # to handle BIDS data
from utils import *  # custom utilities
from pathlib import Path  # to create dirs

import numpy as np
from nilearn.connectome import ConnectivityMeasure

no_session_flag = 4.2  # used to for dummy value of session variable

def main():

    __version__ = get_version()
    msg_info("Version: %s" % __version__)

    args = arguments_manager(__version__)
    fmriprep_dir = get_fmriprep_dir(args)

    msg_info("Indexing BIDS dataset...")

    layout = bidslayout(args.bids_dir, validate=not args.skip_bids_validator)
    layout.add_derivatives(fmriprep_dir)
    subjects_to_analyze = get_subjects_to_analyze(args, layout)
    all_sessions = get_sessions(args, layout)
    space, res = get_space(args, layout)
    task = get_task(args, layout)
    seeds = get_seeds(args)
    denoise_strategies = get_strategies(args)
    layout = setup_output_dir(args, __version__, layout)
    output_dir = layout.derivatives['connectomix'].root

    # print some summary before running
    msg_info("Bids directory: %s" % layout.root)
    msg_info("Fmriprep directory: %s" % fmriprep_dir)
    msg_info("Subject(s) to analyse: %s" % subjects_to_analyze)
    if not res is None:
        msg_info("Selected resolution: %s" % res)

    if args.analysis_level == "participant":

        for subject_label in subjects_to_analyze:

            msg_info("Running for participant %s" % subject_label)

            sessions = []
            for session in all_sessions:
                if session in layout.get_sessions(subject=subject_label):
                    sessions.append(session)
                if session is None:
                    sessions = [None]
            sessions = sorted(sessions)
            if sessions[0] is not None:
                msg_info('Found the following sessions for this subject: %s' % sessions)

            for session in sessions:
                bids_filter = dict(subject=subject_label, return_type='filename',
                              space=space, res=res, task=task, session=session)

                if seeds['type'] == 'all_voxels':
                    seeds['mask'] = get_fmri_mask(layout, bids_filter)

                results = dict()

                for denoise_strategy in denoise_strategies:

                    msg_info('Running denoising strategy %s' % denoise_strategy)

                    # get fmri preprocessed and mask by fmriprep
                    fmri_preproc = get_fmri_preproc(layout, bids_filter, denoise_strategy)

                    results[denoise_strategy] = get_connectivity_measures(fmri_preproc, denoise_strategy, seeds)

                    entities = dict()
                    entities['subject'] = subject_label
                    entities['space'] = space
                    entities['res'] = res
                    entities['session'] = session
                    entities['denoising'] = denoise_strategy
                    outputs = setup_subject_output_paths(layout, entities)

                    # save a couple of nice images
                    item = 'matrix'
                    results[denoise_strategy]['graphics'][item].savefig(outputs[item])

                    if not seeds['type'] == 'all_voxels':
                        item = 'connectome'
                        results[denoise_strategy]['graphics'][item].savefig(outputs[item])

                    # save the matrix data and timeseries
                    for item in ['data', 'timeseries']:
                        np.savetxt(outputs[item], results[denoise_strategy][item], delimiter='\t')

                    build_report(outputs, entities, args, __version__)

    # running group level
    elif args.analysis_level == "group":

        time_series = dict()
        bids_filter = dict(suffix='timeseries', extension='.tsv', return_type='filename')
        msg_warning('Datasets without session not supported yet.')

        group_output_dir = os.path.join(output_dir, 'group')
        Path(group_output_dir).mkdir(parents=True, exist_ok=True)

        for session in layout.derivatives['connectomix'].get_sessions():
            time_series[session] = dict()
            msg_warning('This tool should not be used if you have run several denoising strategies and the outputs are coexisting in the derivative folder.')
            for subject in layout.derivatives['connectomix'].get_subjects(session=session):
                try:
                    series_fn = layout.derivatives['connectomix'].get(**bids_filter, subject=subject, session=session, desc=seeds['type'])[0]
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
