import math
import cv2
import numpy as np
import opFlowOfBlocks as roi


def get_threshold_distance(mag, block_size):
    return mag * block_size


def get_threshold_angle(ang):
    t_angle = float(math.pi) / 2
    return ang + t_angle, ang - t_angle


def get_centre_of_block(blck_1_indx, blck_2_indx, centre_of_blocks):
    x1 = centre_of_blocks[blck_1_indx[0]][blck_1_indx[1]][0]
    y1 = centre_of_blocks[blck_1_indx[0]][blck_1_indx[1]][1]
    x2 = centre_of_blocks[blck_2_indx[0]][blck_2_indx[1]][0]
    y2 = centre_of_blocks[blck_2_indx[0]][blck_2_indx[1]][1]
    slope = float(y2 - y1) / (x2 - x1) if (x1 != x2) else float("inf")
    return (x1, y1), (x2, y2), slope


def calc_euclidean_dist((x1, y1), (x2, y2)):
    dist = float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return dist


def angle_btw_2_blocks(ang1, ang2):
    if ang1 - ang2 < 0:
        ang1_in_deg = math.degrees(ang1)
        ang2_in_deg = math.degrees(ang2)
        return math.radians(360 - (ang1_in_deg - ang2_in_deg))
    return ang1 - ang2


def motion_in_map_generator(op_flow_of_blocks, block_size, centre_of_blocks, x_block_size, y_block_size):
    global frame_no
    motion_inf_val = np.zeros((x_block_size, y_block_size, 8))
    for index, value in np.ndenumerate(op_flow_of_blocks[..., 0]):
        td = get_threshold_distance(op_flow_of_blocks[index[0]][index[1]][0], block_size)
        k = op_flow_of_blocks[index[0]][index[1]][1]
        pos_fi, neg_fi = get_threshold_angle(math.radians(45 * k))

        for ind, val in np.ndenumerate(op_flow_of_blocks[..., 0]):
            if index != ind:
                (x1, y1), (x2, y2), slope = get_centre_of_block(index, ind, centre_of_blocks)
                euclidean_dist = calc_euclidean_dist((x1, y1), (x2, y2))

                if euclidean_dist < td:
                    ang_with_x_axis = math.atan(slope)
                    ang_btw_two_blocks = angle_btw_2_blocks(math.radians(45 * k), ang_with_x_axis)

                    if neg_fi < ang_btw_two_blocks < pos_fi:
                        motion_inf_val[ind[0]][ind[1]][int(op_flow_of_blocks[index[0]][index[1]][1])] += math.exp(
                            -1 * (float(euclidean_dist) / op_flow_of_blocks[index[0]][index[1]][0]))
    frame_no += 1
    return motion_inf_val


def get_motion_infuence_map(vid):
    global frame_no, x_block_size, y_block_size

    frame_no = 0
    cap = cv2.VideoCapture(vid)
    ret, frame1 = cap.read()
    rows, cols = frame1.shape[0], frame1.shape[1]
    print(rows, cols)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    motion_inf_of_frames = []
    count = 0
    while 1:
        print(count)
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        prvs = next_frame
        op_flow_of_blocks, no_of_row_in_block, no_of_col_in_block, block_size, centre_of_blocks, x_block_size, y_block_size = roi.calc_opt_flow_of_blocks(mag, ang, next_frame)
        motion_inf_val = motion_in_map_generator(op_flow_of_blocks, block_size, centre_of_blocks, x_block_size,
                                                 y_block_size)
        motion_inf_of_frames.append(motion_inf_val)

        count += 1
    return motion_inf_of_frames, x_block_size, y_block_size
