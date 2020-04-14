# coding=utf-8
import math
import numpy as np


def calc_opt_flow_of_blocks(mag, angle, grayImg):
    """
        Принимает изображение (ЧБ) и матрицу потока в качестве входных данных.
        Делит изображение на блоки и рассчитывает оптический поток каждого блока
    """
    # рассчитаем количество строк и столбцов в матрице изображения
    rows = grayImg.shape[0]
    cols = grayImg.shape[1]
    no_of_row_in_block = 20
    no_of_col_in_block = 20
    # рассчитаем количество строк блоков и столбцов блоков в кадре
    x_block_size = rows / no_of_row_in_block
    y_block_size = cols / no_of_col_in_block

    op_flow_of_blocks = np.zeros((x_block_size, y_block_size, 2))

    for index, value in np.ndenumerate(mag):
        op_flow_of_blocks[index[0] / no_of_row_in_block][index[1] / no_of_col_in_block][0] += mag[index[0]][index[1]]
        op_flow_of_blocks[index[0] / no_of_row_in_block][index[1] / no_of_col_in_block][1] += angle[index[0]][index[1]]

    centre_of_blocks = np.zeros((x_block_size, y_block_size, 2))
    for index, value in np.ndenumerate(op_flow_of_blocks):
        op_flow_of_blocks[index[0]][index[1]][index[2]] = float(value) / (no_of_row_in_block * no_of_col_in_block)
        val = op_flow_of_blocks[index[0]][index[1]][index[2]]

        if index[2] == 1:
            ang_in_deg = math.degrees(val)
            if ang_in_deg > 337.5:
                k = 0
            else:
                q = ang_in_deg // 22.5
                a1 = q * 22.5
                q1 = ang_in_deg - a1
                a2 = (q + 2) * 22.5
                q2 = a2 - ang_in_deg
                if q1 < q2:
                    k = int(round(a1 / 45))
                else:
                    k = int(round(a2 / 45))
            op_flow_of_blocks[index[0]][index[1]][index[2]] = k

        if index[2] == 0:
            x = ((index[0] + 1) * no_of_row_in_block) - (no_of_row_in_block / 2)
            y = ((index[1] + 1) * no_of_col_in_block) - (no_of_col_in_block / 2)
            centre_of_blocks[index[0]][index[1]][0] = x
            centre_of_blocks[index[0]][index[1]][1] = y
    return op_flow_of_blocks, no_of_row_in_block, no_of_col_in_block, no_of_row_in_block * no_of_col_in_block, centre_of_blocks, x_block_size, y_block_size
