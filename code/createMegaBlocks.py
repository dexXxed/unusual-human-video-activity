import cv2
import numpy as np


def create_mega_blocks(motion_info_of_frames, number_of_rows, number_of_cols):
    n = 2
    mega_block_mot_inf_val = np.zeros(((number_of_rows / n), (number_of_cols / n), len(motion_info_of_frames), 8))

    frame_counter = 0

    for frame in motion_info_of_frames:

        for index, val in np.ndenumerate(frame[..., 0]):
            temp = [list(mega_block_mot_inf_val[index[0] / n][index[1] / n][frame_counter]),
                    list(frame[index[0]][index[1]])]

            mega_block_mot_inf_val[index[0] / n][index[1] / n][frame_counter] = np.array(map(sum, zip(*temp)))

        frame_counter += 1
    print((number_of_rows / n), (number_of_cols / n), len(motion_info_of_frames))
    return mega_block_mot_inf_val


def k_means(mega_block_mot_inf_val):
    cluster_n = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    codewords = np.zeros((len(mega_block_mot_inf_val), len(mega_block_mot_inf_val[0]), cluster_n, 8))

    for row in range(len(mega_block_mot_inf_val)):
        for col in range(len(mega_block_mot_inf_val[row])):
            ret, labels, cw = cv2.kmeans(np.float32(mega_block_mot_inf_val[row][col]),
                                         cluster_n, None, criteria, 10, flags)
            codewords[row][col] = cw

    return codewords
