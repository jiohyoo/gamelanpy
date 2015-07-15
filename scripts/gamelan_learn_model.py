#!/usr/bin/python
__author__ = 'jiohyoo'
import sys
import gamelanpy
import getopt
import numpy as np
import json

def getopts(argv):
    try:
        opts, args = getopt.getopt(argv, "h", [
            "help", "subsample-method=", "subsample-size=",
            "save-model=", "save-weights=", "save-means=", "save-covars=", "save-precs=",
            "l1-depth=", "l1-min=", "l1-max=", "l1-search-repeat=",
            "npn-sample-ratio="])
    except getopt.GetoptError:
        usage()
        raise
        exit(2)

    fit_options = {}
    save_options = {}
    npn = False
    npn_ratio = 0.0

    for opt, arg in opts:
        print opt, arg
        if opt in ['-h', '--help']:
            usage()
            exit()
        elif opt == '--save-model':
            save_options['model'] = arg
        elif opt == '--save-covars':
            save_options['covars'] = arg
        elif opt == '--save-means':
            save_options['means'] = arg
        elif opt == '--save-precs':
            save_options['precs'] = arg
        elif opt == '--save-weights':
            save_options['weights'] = arg
        elif opt == '--l1-depth':
            fit_options['l1_search_depth'] = int(arg)
        elif opt == '--l1-min':
            fit_options['l1_penalty_range_min'] = float(arg)
        elif opt == '--l1-max':
            fit_options['l1_penalty_range_max'] = float(arg)
        elif opt == '--subsample-method':
            fit_options['subsample_method'] = arg
        elif opt == '--subsample-size':
            fit_options['subsample_size'] = int(arg)
        elif opt == '--l1-search-repeat':
            fit_options['l1_search_repeat'] = int(arg)
        elif opt == '--npn-sample-ratio':
            npn = True
            fit_options['npn_sample_ratio'] = float(arg)

    if 'subsample_method' not in fit_options:
        fit_options['subsample_method'] = 'None'
        print 'Warning: Subsampling is not used for learning process.'

    if 'subsample_method' in fit_options \
            and fit_options['subsample_method'] in ['coreset', 'coreset2', 'uniform'] \
            and 'subsample_size' not in fit_options:
        print 'Subsampling method %s is used, but subsample_size is not specified!' % fit_options['subsample_method']
        print 'Please specify the subsample size.'
        exit(2)

    if ('l1_penalty_range_min' in fit_options) is not ('l1_penalty_range_max' in fit_options):
        print 'Please specify both of l1_penalty min and max values.'
        print 'To use default values, do not specify both min and max values.'
        exit(2)

    if npn:
        if fit_options['npn_sample_ratio'] <= 0 or fit_options['npn_sample_ratio'] > 1:
            print 'Nonparanormal sample ratio: %f must be between 0 and 1' % fit_options['npn_sample_ratio']
            exit(2)

    if 'model' not in save_options:
        print 'WARNING: the whole representation of the learned model will not be saved.'

    return fit_options, save_options, npn


def main(data_path, n_components, fit_options, save_options, npn):
    sgmm = gamelanpy.SparseGMM(n_components=n_components, nonparanormal=npn)
    data = np.genfromtxt(data_path, delimiter=',')
    sgmm.fit(data, **fit_options)

    n_vars = sgmm.means_.shape[1]
    if 'model' in save_options:
        json.dump(sgmm, open(save_options['model'], 'w'), cls=gamelanpy.GamelanPyEncoder)
    if 'covars' in save_options:
        np.savetxt(save_options['covars'], sgmm.covars_.reshape((n_components * n_vars, n_vars)), delimiter=',')
    if 'means' in save_options:
        np.savetxt(save_options['means'], sgmm.means_, delimiter=',')
    if 'weights' in save_options:
        np.savetxt(save_options['weights'], sgmm.weights_, delimiter=',')
    if 'precs' in save_options:
        np.savetxt(save_options['precs'], sgmm.precs_.reshape((n_components * n_vars, n_vars)), delimiter=',')


def usage():
    print 'print usage'

if __name__ == '__main__':
    n_components = int(sys.argv[1])
    data_path = sys.argv[2]
    fit_options, save_options, npn = getopts(sys.argv[3:])
    main(data_path, n_components, fit_options, save_options, npn)



