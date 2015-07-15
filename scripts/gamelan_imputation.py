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
            "help", "n-samples="])
    except getopt.GetoptError:
        usage()
        exit(2)

    method = 'predict'
    n_samples = None
    for opt, arg in opts:
        print opt, arg
        if opt in ['-h', '--help']:
            usage()
            exit()
        elif opt == '--n-samples':
            method = 'sample'
            n_samples = int(arg)

    if method == 'sample' and n_samples <= 0:
        print 'Specified number of samples: %d must be positive integer' % n_samples
        exit(2)

    if method not in ['predict', 'sample']:
        print 'Specified method: %s is invalid' % method
        exit(2)

    return method, n_samples


def main(model, data_path, output_path, method, n_samples):

    data = np.genfromtxt(data_path, delimiter=',');
    if method == 'predict':
        output = gamelanpy.predict_missing_values(model, data)
    elif method == 'sample':
        output = gamelanpy.sample_missing_values(model, data, n_samples=n_samples)

    np.savetxt(output_path, output, delimiter=',')


def usage():
    print 'print usage'

if __name__ == '__main__':
    model = json.load(open(sys.argv[1]), object_hook=gamelanpy.gamelan_json_obj_hook)
    data_path = sys.argv[2]
    output_path = sys.argv[3]
    method, n_samples = getopts(sys.argv[4:])
    main(model, data_path, output_path, method, n_samples)



