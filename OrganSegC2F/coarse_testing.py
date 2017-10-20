import numpy as np
import os
import sys
import shutil
import time
import caffe
from utils import *


data_path = sys.argv[1]
current_fold = int(sys.argv[2])
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
slice_thickness = int(sys.argv[6])
organ_ID = int(sys.argv[7])
plane = sys.argv[8]
GPU_ID = int(sys.argv[9])
learning_rate = float(sys.argv[10])
gamma = float(sys.argv[11])
snapshot_path = os.path.join(snapshot_path, 'coarse:' + sys.argv[10])
result_path = os.path.join(result_path, 'coarse:' + sys.argv[10])
starting_iterations = int(sys.argv[12])
step = int(sys.argv[13])
max_iterations = int(sys.argv[14])
iteration = range(starting_iterations, max_iterations + 1, step)
timestamp = sys.argv[15]
snapshot_name = snapshot_name_from_timestamp(snapshot_path, \
    current_fold, plane, 'C', slice_thickness, organ_ID, iteration, timestamp)
if snapshot_name == '':
    exit('Error: no valid snapshot directory is detected!')
snapshot_directory = os.path.join(snapshot_path, snapshot_name)
print 'Snapshot directory: ' + snapshot_directory + ' .'
snapshot = []
for t in range(len(iteration)):
    snapshot_file = snapshot_filename(snapshot_directory, iteration[t])
    snapshot.append(snapshot_file)
print str(len(snapshot)) + ' snapshots are to be evaluated.'
for t in range(len(snapshot)):
    print '  Snapshot #' + str(t + 1) + ': ' + snapshot[t] + ' .'
result_name = snapshot_name

sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()

volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()
DSC = np.zeros((len(snapshot), len(volume_list)))
result_directory = os.path.join(result_path, result_name, 'volumes')
if not os.path.exists(result_directory):
    os.makedirs(result_directory)
result_file = os.path.join(result_path, result_name, 'results.txt')
output = open(result_file, 'w')
output.close()
for t in range(len(snapshot)):
    output = open(result_file, 'a+')
    output.write('Evaluating snapshot ' + str(iteration[t]) + ':\n')
    output.close()
    finished = True
    for i in range(len(volume_list)):
        volume_file = volume_filename_testing(result_directory, iteration[t], i)
        if not os.path.isfile(volume_file):
            finished = False
            break
    if not finished:
        deploy_filename = 'deploy_' + str(slice_thickness) + '.prototxt'
        deploy_file = os.path.join(prototxt_path, deploy_filename)
        if not os.path.isfile(deploy_file):
            deploy_file_ = os.path.join('prototxts', deploy_filename)
            shutil.copyfile(deploy_file_, deploy_file)
        net = caffe.Net(deploy_file, snapshot[t], caffe.TEST)
    for i in range(len(volume_list)):
        start_time = time.time()
        print 'Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases, ' + \
            str(t + 1) + ' out of ' + str(len(snapshot)) + ' snapshots.'
        volume_file = volume_filename_testing(result_directory, iteration[t], i)
        s = volume_list[i].split(' ')
        label = np.load(s[2])
        label = is_organ(label, organ_ID).astype(np.uint8)
        if not os.path.isfile(volume_file):
            image = np.load(s[1])
            print '  Data loading is finished: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred = np.zeros_like(image, dtype = np.float32)
            minR = 0
            if plane == 'X':
                maxR = label.shape[0]
            elif plane == 'Y':
                maxR = label.shape[1]
            elif plane == 'Z':
                maxR = label.shape[2]
            for j in range(minR, maxR):
                if slice_thickness == 1:
                    sID = [j, j, j]
                elif slice_thickness == 3:
                    sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                if plane == 'X':
                    image_ = image[sID, :, :].astype(np.float32)
                    label_ = label[sID, :, :].astype(np.uint8)
                elif plane == 'Y':
                    image_ = image[:, sID, :].transpose(1, 0, 2).astype(np.float32)
                    label_ = label[:, sID, :].transpose(1, 0, 2).astype(np.uint8)
                elif plane == 'Z':
                    image_ = image[:, :, sID].transpose(2, 0, 1).astype(np.float32)
                    label_ = label[:, :, sID].transpose(2, 0, 1).astype(np.uint8)
                image_[image_ > high_range] = high_range
                image_[image_ < low_range] = low_range
                image_ = (image_ - low_range) / (high_range - low_range)
                image_ = image_.reshape((1, 3, image_.shape[1], image_.shape[2]))
                net.blobs['data'].reshape(*image_.shape)
                net.blobs['data'].data[...] = image_
                net.forward()
                out = net.blobs['prob'].data[0, :, :, :]
                if slice_thickness == 1:
                    if plane == 'X':
                        pred[j, :, :] = out
                    elif plane == 'Y':
                        pred[:, j, :] = out
                    elif plane == 'Z':
                        pred[:, :, j] = out
                elif slice_thickness == 3:
                    if plane == 'X':
                        pred[max(minR, j - 1): min(maxR, j + 2), :, :] += \
                            out[max(0, 1 - j): min(3, maxR + 1 - j), :, :]
                    elif plane == 'Y':
                        pred[:, max(minR, j - 1): min(maxR, j + 2), :] += \
                            out[max(0, 1 - j): min(3, maxR + 1 - j), :, :].transpose(1, 0, 2)
                    elif plane == 'Z':
                        pred[:, :, max(minR, j - 1): min(maxR, j + 2)] += \
                            out[max(0, 1 - j): min(3, maxR + 1 - j), :, :].transpose(1, 2, 0)
            if slice_thickness == 3:
                if plane == 'X':
                    pred[minR, :, :] /= 2
                    pred[minR + 1: maxR - 1, :, :] /= 3
                    pred[maxR - 1, :, :] /= 2
                elif plane == 'Y':
                    pred[:, minR, :] /= 2
                    pred[:, minR + 1: maxR - 1, :] /= 3
                    pred[:, maxR - 1, :] /= 2
                elif plane == 'Z':
                    pred[:, :, minR] /= 2
                    pred[:, :, minR + 1: maxR - 1] /= 3
                    pred[:, :, maxR - 1] /= 2
            print '  Testing is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.'
            pred = np.uint8(np.around(pred * 255))
            np.savez_compressed(volume_file, volume = pred)
            print '  Data saving is finished: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred_temp = np.zeros_like(pred, dtype = np.bool)
            pred_temp[pred >= 128] = True
        else:
            volume_data = np.load(volume_file)
            pred = volume_data['volume']
            print '  Testing result is loaded: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
            pred_temp = np.zeros_like(pred, dtype = np.bool)
            pred_temp[pred >= 128] = True
        DSC[t, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred_temp)
        print '    DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + \
            ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .'
        output = open(result_file, 'a+')
        output.write('  Testcase ' + str(i + 1) + ': DSC = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[t, i]) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC[t, i] = 0
        print '  DSC computation is finished: ' + \
            str(time.time() - start_time) + ' second(s) elapsed.'
    print 'Snapshot ' + str(iteration[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .'
    output = open(result_file, 'a+')
    output.write('Snapshot ' + str(iteration[t]) + \
        ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .\n')
    output.close()
print 'The testing process is finished.'
for t in range(len(snapshot)):
    print '  Snapshot ' + str(iteration[t]) + ': average DSC = ' + str(np.mean(DSC[t, :])) + ' .'
