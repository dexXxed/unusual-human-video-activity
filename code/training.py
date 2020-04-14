# coding=utf-8
import numpy as np
import createMegaBlocks as cmb
import motionInfuenceGenerator as mig


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def train_from_video(vid):
    """
        Вызывает все методы для обучения из заданного видео
        Должен возвращать кодовые слова или их хранить
    """
    print "Training From ", vid
    motion_inf_of_frames, rows, cols = mig.get_motion_infuence_map(vid)
    print "Motion Inf Map", len(motion_inf_of_frames)
    mega_block_mot_inf_val = cmb.create_mega_blocks(motion_inf_of_frames, rows, cols)
    np.save("..\Dataset\\videos\scene1\megaBlockMotInfVal_set1_p1_train_40-40_k5.npy", mega_block_mot_inf_val)
    print(np.amax(mega_block_mot_inf_val))
    print(np.amax(reject_outliers(mega_block_mot_inf_val)))

    codewords = cmb.k_means(mega_block_mot_inf_val)
    np.save("..\Dataset\\videos\scene1\codewords_set1_p1_train_40-40_k5.npy", codewords)
    print codewords
    return


if __name__ == '__main__':
    '''
        определяет обучающий датасет и вызывает train_from_video для каждого видео
    '''
    trainingSet = [r"..\Dataset\\videos\\scene1\\train1.avi"]
    for video in trainingSet:
        train_from_video(video)
    print "Done"
