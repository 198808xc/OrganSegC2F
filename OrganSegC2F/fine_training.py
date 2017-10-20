import numpy as np
import os
import sys
import shutil
import urllib
import caffe
from utils import *
import fine_surgery


data_path = sys.argv[1]
current_fold = sys.argv[2]
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
slice_threshold = float(sys.argv[6])
slice_thickness = int(sys.argv[7])
training_margin = int(sys.argv[8])
random_prob = float(sys.argv[9])
sample_batch = int(sys.argv[10])
organ_ID = int(sys.argv[11])
plane = sys.argv[12]
GPU_ID = int(sys.argv[13])
learning_rate = float(sys.argv[14])
gamma = float(sys.argv[15])
snapshot_path = os.path.join(snapshot_path, 'fine:' + sys.argv[14] + ',' + str(training_margin))
starting_iterations = int(sys.argv[16])
step = int(sys.argv[17])
max_iterations = [int(s) for s in sys.argv[18].split(',')]
timestamp = sys.argv[19]
snapshot_name = 'FD' + str(current_fold) + ':' + \
    plane + 'F' + str(slice_thickness) + '_' + str(organ_ID) + '_' + timestamp
log_file = os.path.join(log_path, snapshot_name + '.txt')
snapshot_directory = os.path.join(snapshot_path, snapshot_name)
if not os.path.exists(snapshot_directory):
    os.makedirs(snapshot_directory)
log_file_ = log_filename(snapshot_directory)
weights = os.path.join(pretrained_model_path, 'fcn8s-heavy-pascal.caffemodel')
if not os.path.isfile(weights):
    print 'Downloading <' + weights + '> from the Internet ...'
    urllib.urlretrieve('http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel', weights)

if __name__ == '__main__':
    solver_filename = 'solver_F' + str(slice_thickness) + '.prototxt'
    solver_file = os.path.join(prototxt_path, solver_filename)
    output = open(solver_file, 'w')
    prototxt_filename = 'training_F' + str(slice_thickness) + '.prototxt'
    prototxt_file = os.path.join(prototxt_path, prototxt_filename)
    if not os.path.isfile(prototxt_file):
        prototxt_file_ = os.path.join('prototxts', prototxt_filename)
        shutil.copyfile(prototxt_file_, prototxt_file)
    output.write('train_net: \"' + prototxt_file + '\"\n')
    output.write('\n' * 1)
    output.write('display: 20\n')
    output.write('average_loss: 20\n')
    output.write('\n' * 1)
    output.write('base_lr: ' + str(learning_rate) + '\n')
    output.write('lr_policy: \"multistep\"\n')
    output.write('gamma: ' + str(gamma) + '\n')
    for t in range(len(max_iterations) - 1):
        output.write('stepvalue: ' + str(max_iterations[t]) + '\n')
    output.write('max_iter: ' + str(max_iterations[-1]) + '\n')
    output.write('\n' * 1)
    output.write('momentum: 0.99\n')
    output.write('\n' * 1)
    output.write('iter_size: 1\n')
    output.write('weight_decay: 0.0005\n')
    output.write('snapshot: ' + str(step) + '\n')
    output.write('snapshot_prefix: \"' + os.path.join(snapshot_directory, 'train') + '\"\n')
    output.write('\n' * 1)
    output.write('test_initialization: false\n')
    output.close()
    sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
    caffe.set_device(GPU_ID)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_file)
    if starting_iterations == 0:
        solver.net.copy_from(weights)
    else:
        snapshot_ = os.path.join( \
            snapshot_directory, 'train_iter_' + str(starting_iterations) + '.solverstate')
        solver.restore(snapshot_)
    interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
    fine_surgery.interp(solver.net, interp_layers)
    solver.step(max_iterations[-1])
    shutil.copyfile(log_file, log_file_)
