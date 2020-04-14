# coding=utf-8
import cv2
import numpy as np

import createMegaBlocks as cmb
import motionInfuenceGenerator as mig


def square(a):
    return a ** 2


def diff(l):
    return l[0] - l[1]


def show_unusual_activities(unusual, vid, noOfRows, noOfCols, n):
    unusual_frames = unusual.keys()
    unusual_frames.sort()
    print(unusual_frames)
    cap = cv2.VideoCapture(vid)
    ret, frame = cap.read()
    rows, cols = frame.shape[0], frame.shape[1]
    row_length = rows / (noOfRows / n)
    col_length = cols / (noOfCols / n)
    print("Block Size ", (row_length, col_length))
    count = 0
    screen_res = 980, 520
    scale_width = screen_res[0] / 320
    scale_height = screen_res[1] / 240
    scale = min(scale_width, scale_height)
    window_width = int(320 * scale)
    window_height = int(240 * scale)

    cv2.namedWindow('Unusual Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Unusual Frame', window_width, window_height)
    while 1:
        print(count)
        ret, u_frame = cap.read()

        if count in unusual_frames:
            if not ret:
                break
            for blockNum in unusual[count]:
                print(blockNum)
                x1 = blockNum[1] * row_length
                y1 = blockNum[0] * col_length
                x2 = (blockNum[1] + 1) * row_length
                y2 = (blockNum[0] + 1) * col_length
                cv2.rectangle(u_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
            print("Unusual frame number ", str(count))
        cv2.imshow('Unusual Frame', u_frame)

        cv2.waitKey(0)
        count += 1


def construct_min_dist_matrix(megaBlockMotInfVal, codewords, noOfRows, noOfCols, vid):
    threshold = 5.83682407063e-05
    n = 2
    min_dist_matrix = np.zeros((len(megaBlockMotInfVal[0][0]), (noOfRows / n), (noOfCols / n)))
    for index, val in np.ndenumerate(megaBlockMotInfVal[..., 0]):
        eucledian_dist = []
        for codeword in codewords[index[0]][index[1]]:
            temp = [list(megaBlockMotInfVal[index[0]][index[1]][index[2]]), list(codeword)]
            dist = np.linalg.norm(megaBlockMotInfVal[index[0]][index[1]][index[2]] - codeword)
            euc_dist = (sum(map(square, map(diff, zip(*temp))))) ** 0.5
            eucledian_dist.append(euc_dist)
        min_dist_matrix[index[2]][index[0]][index[1]] = min(eucledian_dist)
    unusual = {}
    for i in range(len(min_dist_matrix)):
        if np.amax(min_dist_matrix[i]) > threshold:
            unusual[i] = []
            for index, val in np.ndenumerate(min_dist_matrix[i]):
                if val > threshold:
                    unusual[i].append((index[0], index[1]))
    print(unusual)
    show_unusual_activities(unusual, vid, noOfRows, noOfCols, n)


def test_video(vid):
    print("Test video ", vid)
    motion_inf_of_frames, rows, cols = mig.get_motion_infuence_map(vid)

    mega_block_mot_inf_val = cmb.create_mega_blocks(motion_inf_of_frames, rows, cols)
    np.save("..\\Dataset\\videos\scene1\megaBlockMotInfVal_set1_p1_test_20-20_k5.npy", mega_block_mot_inf_val)
    codewords = np.load("..\\Dataset\\videos\scene1\codewords_set2_p1_train_20-20_k5.npy")
    print("codewords", codewords)
    return


if __name__ == '__main__':
    '''
        определяет датасет для тестирования и вызывает test_video для каждого видео
    '''
    testSet = [r"..\Dataset\videos\scene1\test1.avi"]
    for video in testSet:
        test_video(video)
    print("Done")
