import numpy as np
import os
import sys
import shutil
import time
import caffe
from utils import *
import scipy.io


data_path = sys.argv[1]
current_fold = int(sys.argv[2])
organ_number = int(sys.argv[3])
low_range = int(sys.argv[4])
high_range = int(sys.argv[5])
organ_ID = int(sys.argv[6])
volume_list = open(testing_set_filename(current_fold), 'r').read().splitlines()
while volume_list[len(volume_list) - 1] == '':
    volume_list.pop()
GPU_ID = int(sys.argv[7])
coarse_slice_thickness = int(sys.argv[8])
coarse_learning_rate = float(sys.argv[9])
coarse_gamma = float(sys.argv[10])
coarse_snapshot_path = os.path.join(snapshot_path, 'coarse:' + sys.argv[9])
coarse_result_path = os.path.join(result_path, 'coarse:' + sys.argv[9])
coarse_starting_iterations = int(sys.argv[11])
coarse_step = int(sys.argv[12])
coarse_max_iterations = int(sys.argv[13])
coarse_iteration = range(coarse_starting_iterations, coarse_max_iterations + 1, coarse_step)
coarse_threshold = float(sys.argv[14])
coarse_timestamp = {}
coarse_timestamp['X'] = sys.argv[15]
coarse_timestamp['Y'] = sys.argv[16]
coarse_timestamp['Z'] = sys.argv[17]
coarse_result_name_ = {}
coarse_result_directory_ = {}
for plane in ['X', 'Y', 'Z']:
    coarse_result_name = result_name_from_timestamp(coarse_result_path, current_fold, \
        plane, 'C', coarse_slice_thickness, \
        organ_ID, coarse_iteration, volume_list, coarse_timestamp[plane])
    if coarse_result_name == '':
        exit('Error: no valid result directory is detected!')
    coarse_result_directory = os.path.join(coarse_result_path, coarse_result_name, 'volumes')
    print 'Coarse-scaled result directory for plane ' + \
        plane + ': ' + coarse_result_directory + ' .'
    if coarse_result_name.startswith('FD'):
        index_ = coarse_result_name.find(':')
        coarse_result_name = coarse_result_name[index_ + 1: ]
    coarse_result_name_[plane] = coarse_result_name
    coarse_result_directory_[plane] = coarse_result_directory
coarse_result_name = 'FD' + str(current_fold) + ':' + 'fusion:' + \
    coarse_result_name_['X'] + ',' + coarse_result_name_['Y'] + ',' + \
    coarse_result_name_['Z'] + ',' + str(coarse_starting_iterations) + '_' + \
    str(coarse_step) + '_' + str(coarse_max_iterations) + ',' + \
    str(coarse_threshold)
coarse_result_directory = os.path.join(coarse_result_path, coarse_result_name, 'volumes')

coarse_fusion_code = sys.argv[18]
fine_slice_thickness = int(sys.argv[19])
training_margin = int(sys.argv[20])
fine_learning_rate = float(sys.argv[21])
fine_gamma = float(sys.argv[22])
fine_snapshot_path = os.path.join(snapshot_path, \
    'fine:' + sys.argv[21] + ',' + str(training_margin))
fine_result_path = os.path.join(result_path, 'fine:' + sys.argv[21] + ',' + str(training_margin))
fine_starting_iterations = int(sys.argv[23])
fine_step = int(sys.argv[24])
fine_max_iterations = int(sys.argv[25])
fine_iteration = range(fine_starting_iterations, fine_max_iterations + 1, fine_step)
fine_threshold = float(sys.argv[26])
fine_timestamp = {}
fine_timestamp['X'] = sys.argv[27]
fine_timestamp['Y'] = sys.argv[28]
fine_timestamp['Z'] = sys.argv[29]
fine_snapshot_name_ = {}
fine_snapshot_directory_ = {}
fine_snapshot_ = {}
for plane in ['X', 'Y', 'Z']:
    fine_snapshot_name = snapshot_name_from_timestamp(fine_snapshot_path, current_fold, \
        plane, 'F', fine_slice_thickness, organ_ID, fine_iteration, fine_timestamp[plane])
    if fine_snapshot_name == '':
        exit('Error: no valid snapshot directory is detected!')
    fine_snapshot_directory = os.path.join(fine_snapshot_path, fine_snapshot_name)
    print 'Snapshot directory: ' + fine_snapshot_directory + ' .'
    fine_snapshot = []
    for t in range(len(fine_iteration)):
        fine_snapshot_file = snapshot_filename(fine_snapshot_directory, fine_iteration[t])
        fine_snapshot.append(fine_snapshot_file)
    print str(len(fine_snapshot)) + ' snapshots are to be evaluated.'
    for s in range(len(fine_snapshot)):
        print '  Snapshot #' + str(s + 1) + ': ' + fine_snapshot[s] + ' .'
    if fine_snapshot_name.startswith('FD'):
        index_ = fine_snapshot_name.find(':')
        fine_snapshot_name = fine_snapshot_name[index_ + 1: ]
    fine_snapshot_name_[plane] = fine_snapshot_name
    fine_snapshot_directory_[plane] = fine_snapshot_directory
    fine_snapshot_[plane] = fine_snapshot
testing_margin = int(sys.argv[30])
max_rounds = int(sys.argv[31])
coarse2fine_result_path = os.path.join(result_path, \
    'coarse2fine:' + sys.argv[9] + ',' + sys.argv[21] + ',' + str(training_margin))
result_name = 'FD' + str(current_fold) + ':' + \
    'coarse:' + coarse_result_name_['X'] + ',' + coarse_result_name_['Y'] + ',' + \
    coarse_result_name_['Z'] + ',' + str(coarse_starting_iterations) + '_' + \
    str(coarse_step) + '_' + str(coarse_max_iterations) + ',' + \
    str(coarse_threshold) + ',' + coarse_fusion_code + ':' + \
    'fine:' + fine_snapshot_name_['X'] + ',' + \
    fine_snapshot_name_['Y'] + ',' + fine_snapshot_name_['Z'] + ',' + \
    str(fine_starting_iterations) + '_' + str(fine_step) + '_' + str(fine_max_iterations) + ',' + \
    str(testing_margin) + ',' + str(fine_threshold)
coarse2fine_result_directory = os.path.join(coarse2fine_result_path, result_name, 'volumes')

finished = np.ones((len(volume_list)), dtype = np.int)
for i in range(len(volume_list)):
    for r in range(max_rounds + 1):
        volume_file = volume_filename_coarse2fine(coarse2fine_result_directory, r, i)
        if not os.path.isfile(volume_file):
            finished[i] = 0
            break
finished_all = (finished.sum() == len(volume_list))
if finished_all:
    exit()
else:
    deploy_filename = 'deploy_' + str(fine_slice_thickness) + '.prototxt'
    deploy_file = os.path.join(prototxt_path, deploy_filename)
    if not os.path.isfile(deploy_file):
        deploy_file_ = os.path.join('prototxts', deploy_filename)
        shutil.copyfile(deploy_file_, deploy_file)

sys.path.insert(0, os.path.join(CAFFE_root, 'python'))
caffe.set_device(GPU_ID)
caffe.set_mode_gpu()
net_ = {}
for plane in ['X', 'Y', 'Z']:
    net_[plane] = []
    for t in range(len(fine_iteration)):
        net = caffe.Net(deploy_file, fine_snapshot_[plane][t], caffe.TEST)
        net_[plane].append(net)

DSC = np.zeros((max_rounds + 1, len(volume_list)))
DSC_90 = np.zeros((len(volume_list)))
DSC_95 = np.zeros((len(volume_list)))
DSC_98 = np.zeros((len(volume_list)))
DSC_99 = np.zeros((len(volume_list)))
result_directory = os.path.join(coarse2fine_result_path, result_name, 'volumes')
if not os.path.exists(coarse2fine_result_directory):
    os.makedirs(coarse2fine_result_directory)
result_file = os.path.join(coarse2fine_result_path, result_name, 'results.txt')
output = open(result_file, 'w')
output.close()
output = open(result_file, 'a+')
output.write('Fusing results of ' + str(len(coarse_iteration)) + \
    ' and ' + str(len(fine_iteration)) + ' snapshots:\n')
output.close()
for i in range(len(volume_list)):
    start_time = time.time()
    print 'Testing ' + str(i + 1) + ' out of ' + str(len(volume_list)) + ' testcases.'
    output = open(result_file, 'a+')
    output.write('  Testcase ' + str(i + 1) + ':\n')
    output.close()
    s = volume_list[i].split(' ')
    label = np.load(s[2])
    label = is_organ(label, organ_ID).astype(np.uint8)
    if not finished[i]:
        image = np.load(s[1])
    print '  Data loading is finished: ' + str(time.time() - start_time) + ' second(s) elapsed.'
    terminated = False
    for r in range(max_rounds + 1):
        print '  Iteration round ' + str(r) + ':'
        volume_file = volume_filename_coarse2fine(coarse2fine_result_directory, r, i)
        if terminated or not os.path.isfile(volume_file):
            terminated = True
            pred = np.zeros_like(label, dtype = np.int8)
            if r == 0:
                coarse_volume_file = volume_filename_fusion( \
                    coarse_result_directory, coarse_fusion_code, i)
                volume_data = np.load(coarse_volume_file)
                pred = volume_data['volume']
                print '    Fusion is finished: ' + \
                    str(time.time() - start_time) + ' second(s) elapsed.'
            else:
                if pred_prev.sum() == 0:
                    continue
                pred_ = np.zeros_like(label, dtype = np.float32)
                for plane in ['X', 'Y', 'Z']:
                    for t in range(len(fine_iteration)):
                        pred__ = np.zeros_like(image, dtype = np.float32)
                        net = net_[plane][t]
                        minR = 0
                        if plane == 'X':
                            maxR = label.shape[0]
                        elif plane == 'Y':
                            maxR = label.shape[1]
                        elif plane == 'Z':
                            maxR = label.shape[2]
                        for j in range(minR, maxR):
                            if fine_slice_thickness == 1:
                                sID = [j, j, j]
                            elif fine_slice_thickness == 3:
                                sID = [max(minR, j - 1), j, min(maxR - 1, j + 1)]
                            if plane == 'X':
                                image_ = image[sID, :, :].astype(np.float32)
                                label_ = pred_prev[sID, :, :].astype(np.uint8)
                            elif plane == 'Y':
                                image_ = image[:, sID, :].transpose(1, 0, 2).astype(np.float32)
                                label_ = pred_prev[:, sID, :].transpose(1, 0, 2).astype(np.uint8)
                            elif plane == 'Z':
                                image_ = image[:, :, sID].transpose(2, 0, 1).astype(np.float32)
                                label_ = pred_prev[:, :, sID].transpose(2, 0, 1).astype(np.uint8)
                            if label_.sum() == 0:
                                continue
                            width = label_.shape[1]
                            height = label_.shape[2]
                            arr = np.nonzero(label_)
                            minA = min(arr[1])
                            maxA = max(arr[1])
                            minB = min(arr[2])
                            maxB = max(arr[2])
                            image_ = image_[:, max(minA - testing_margin, 0): \
                                min(maxA + testing_margin + 1, width), \
                                max(minB - testing_margin, 0): \
                                min(maxB + testing_margin + 1, height)]
                            image_[image_ > high_range] = high_range
                            image_[image_ < low_range] = low_range
                            image_ = (image_ - low_range) / (high_range - low_range)
                            image_ = image_.reshape(1, 3, image_.shape[1], image_.shape[2])
                            net.blobs['data'].reshape(*image_.shape)
                            net.blobs['data'].data[...] = image_
                            net.forward()
                            out = net.blobs['prob'].data[0, :, :, :]
                            if fine_slice_thickness == 1:
                                if plane == 'X':
                                    pred__[j, max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height)] = out
                                elif plane == 'Y':
                                    pred__[max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), j, \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height)] = out
                                elif plane == 'Z':
                                    pred__[max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height), j] = out
                            elif fine_slice_thickness == 3:
                                if plane == 'X':
                                    pred__[max(minR, j - 1): min(maxR, j + 2), \
                                        max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height)] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), :, :]
                                elif plane == 'Y':
                                    pred__[max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minR, j - 1): min(maxR, j + 2), \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height)] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), \
                                            :, :].transpose(1, 0, 2)
                                elif plane == 'Z':
                                    pred__[max(minA - testing_margin, 0): \
                                        min(maxA + testing_margin + 1, width), \
                                        max(minB - testing_margin, 0): \
                                        min(maxB + testing_margin + 1, height), \
                                        max(minR, j - 1): min(maxR, j + 2)] += \
                                        out[max(minR, 1 - j): min(3, maxR + 1 - j), \
                                            :, :].transpose(1, 2, 0)
                        if fine_slice_thickness == 3:
                            if plane == 'X':
                                pred__[minR, :, :] /= 2
                                pred__[minR + 1: maxR - 1, :, :] /= 3
                                pred__[maxR - 1, :, :] /= 2
                            elif plane == 'Y':
                                pred__[:, minR, :] /= 2
                                pred__[:, minR + 1: maxR - 1, :] /= 3
                                pred__[:, maxR - 1, :] /= 2
                            elif plane == 'Z':
                                pred__[:, :, minR] /= 2
                                pred__[:, :, minR + 1: maxR - 1] /= 3
                                pred__[:, :, maxR - 1] /= 2
                        print '    Testing on plane ' + plane + ' and snapshot ' + str(t + 1) + \
                            ' is finished: ' + str(time.time() - start_time) + \
                            ' second(s) elapsed.'
                        pred_ = pred_ + pred__
                pred[pred_ >= fine_threshold * 3 * len(fine_iteration)] = 1
                pred = post_processing(pred, pred, 0.5, organ_ID)
                print '    Testing is finished: ' + \
                    str(time.time() - start_time) + ' second(s) elapsed.'
            np.savez_compressed(volume_file, volume = pred)
        else:
            volume_data = np.load(volume_file)
            pred = volume_data['volume']
            print '    Testing result is loaded: ' + \
                str(time.time() - start_time) + ' second(s) elapsed.'
        DSC[r, i], inter_sum, pred_sum, label_sum = DSC_computation(label, pred)
        print '      DSC = 2 * ' + str(inter_sum) + ' / (' + str(pred_sum) + ' + ' + \
            str(label_sum) + ') = ' + str(DSC[r, i]) + ' .'
        output = open(result_file, 'a+')
        output.write('    Round ' + str(r) + ', ' + 'DSC = 2 * ' + str(inter_sum) + ' / (' + \
            str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(DSC[r, i]) + ' .\n')
        output.close()
        if pred_sum == 0 and label_sum == 0:
            DSC[r, i] = 0
        if r > 0:
            inter_DSC, inter_sum, pred_sum, label_sum = DSC_computation(pred_prev, pred)
            if pred_sum == 0 and label_sum == 0:
                inter_DSC = 1
            print '        Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .'
            output = open(result_file, 'a+')
            output.write('      Inter-iteration DSC = 2 * ' + str(inter_sum) + ' / (' + \
                str(pred_sum) + ' + ' + str(label_sum) + ') = ' + str(inter_DSC) + ' .\n')
            output.close()
            if DSC_90[i] == 0 and (r == max_rounds or inter_DSC >= 0.90):
                DSC_90[i] = DSC[r, i]
            if DSC_95[i] == 0 and (r == max_rounds or inter_DSC >= 0.95):
                DSC_95[i] = DSC[r, i]
            if DSC_98[i] == 0 and (r == max_rounds or inter_DSC >= 0.98):
                DSC_98[i] = DSC[r, i]
            if DSC_99[i] == 0 and (r == max_rounds or inter_DSC >= 0.99):
                DSC_99[i] = DSC[r, i]
        if r <= max_rounds:
            pred_prev = np.copy(pred)
for r in range(max_rounds + 1):
    print 'Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .'
    output = open(result_file, 'a+')
    output.write('Round ' + str(r) + ', ' + 'Average DSC = ' + str(np.mean(DSC[r, :])) + ' .\n')
    output.close()
print 'DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .'
print 'DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .'
print 'DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .'
print 'DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .'
output = open(result_file, 'a+')
output.write('DSC threshold = 0.90, ' + 'Average DSC = ' + str(np.mean(DSC_90)) + ' .\n')
output.write('DSC threshold = 0.95, ' + 'Average DSC = ' + str(np.mean(DSC_95)) + ' .\n')
output.write('DSC threshold = 0.98, ' + 'Average DSC = ' + str(np.mean(DSC_98)) + ' .\n')
output.write('DSC threshold = 0.99, ' + 'Average DSC = ' + str(np.mean(DSC_99)) + ' .\n')
output.close()
print 'The coarse-to-fine testing process is finished.'
