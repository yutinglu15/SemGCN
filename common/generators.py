from __future__ import print_function, absolute_import

from functools import reduce

import numpy as np
import torch
from torch.utils.data import Dataset


class PoseGenerator(Dataset):
    # def __init__(self, poses_3d, poses_2d, actions, opt='train'):
    def __init__(self, opt='train'):
        # assert poses_3d is not None

        # self._poses_3d = np.concatenate(poses_3d)
        # self._poses_2d = np.concatenate(poses_2d)
        piece = self.data_load('D:/projects/hackathon/Hackathon-GPA/data/', option=opt)
        self._poses_2d = piece['X']

        self._poses_3d = np.zeros((piece['Y'].shape[0], 16, 3))
        self._poses_3d[:, :6, :] = piece['Y'][:, :6, :]
        self._poses_3d[:, 7:, :] = piece['Y'][:, 6:, :]

        # self._actions = reduce(lambda x, y: x + y, actions)

        print('2d shape: ', self._poses_2d.shape)
        print('3d shape: ', self._poses_3d.shape)
        # print('action shape: ', len(self._actions))

        # assert self._poses_3d.shape[0] == self._poses_2d.shape[0] and self._poses_3d.shape[0] == len(self._actions)
        # print('Generating {} poses...'.format(len(self._actions)))

    def data_load(self, my_dir, load_gf=False, option='train test val'):
        train = {}
        val = {}
        test = {}

        #     train['X'] = np.load('train/gt_2d_train.npy')- np.load('train/train_2d_gt_mean.npy').reshape((16,2))
        #     train['X'] /= np.load('train/train_2d_gt_std.npy').reshape((16,2))
        options = option.split()
        if 'train' in options:
            train['X'] = np.load(my_dir + 'train/gt_2d_train.npy')

            train['Y'] = np.load(my_dir + 'train/gt_3d_train_rel.npy')


        #     val['X'] = np.load('validation/gt_2d_val.npy')- np.load('validation/val_2d_gt_mean.npy').reshape((16,2))
        #     val['X'] /= np.load('validation/val_2d_gt_std.npy').reshape((16,2))
        if 'val' in options:
            val['X'] = np.load(my_dir + 'validation/gt_2d_val.npy')

            val['Y'] = np.load(my_dir + 'validation/gt_3d_val_rel.npy')

        if 'test' in options:
            test['X'] = np.load(my_dir + 'test/gt_2d_test.npy')

        #     test['Y'] = np.load('test/gt_3d_test_rel.npy')

        if load_gf:
            train["X_gt_mdp_15"] = np.load(my_dir + 'train/train_gt_mdp_15.npy')
            train["J_gt_mdp_15"] = np.load(my_dir + 'train/J_train_gt_mdp_15.npy')
            train["J_gt_mdp_enc2_15"] = np.load(my_dir + 'train/J_train_gt_mdp_enc2_15.npy')

            val["X_gt_mdp_15"] = np.load(my_dir + 'validation/val_gt_mdp_15.npy')
            val["J_gt_mdp_15"] = np.load(my_dir + 'validation/J_val_gt_mdp_15.npy')
            val["J_gt_mdp_enc2_15"] = np.load(my_dir + 'validation/J_val_gt_mdp_enc2_15.npy')

            test["X_gt_mdp_15"] = np.load(my_dir + 'test/test_gt_mdp_15.npy')
            test["J_gt_mdp_15"] = np.load(my_dir + 'test/J_test_gt_mdp_15.npy')
            test["J_gt_mdp_enc2_15"] = np.load(my_dir + 'test/J_test_gt_mdp_enc2_15.npy')

        if option == 'train':
            return train

        elif option == 'val':
            return val

        else:
            return train, val, test


    def __getitem__(self, index):
        out_pose_3d = self._poses_3d[index]
        out_pose_2d = self._poses_2d[index]
        # out_action = self._actions[index]

        out_pose_3d = torch.from_numpy(out_pose_3d).float()
        out_pose_2d = torch.from_numpy(out_pose_2d).float()

        return out_pose_3d, out_pose_2d
        # return out_pose_3d, out_pose_2d, out_action

    def __len__(self):
        return self._poses_2d.shape[0]
        # return len(self._actions)
