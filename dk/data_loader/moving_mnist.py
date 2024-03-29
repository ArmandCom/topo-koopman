import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data
from scipy.spatial.transform import Rotation as Ro
# from utils.plots import plot_3Dcurve, plot_2Dcurve
import cv2

def generate_random_chunk(full_length, chunk_length):
    start = np.random.randint(low = 1, high = full_length-chunk_length)
    seq = np.arange(start, start+chunk_length)
    return seq

def load_mnist(root):
    # Load MNIST dataset for generating training data.
    path = os.path.join(root, 'moving_mnist/train-images-idx3-ubyte.gz')
    with gzip.open(path, 'rb') as f:
        mnist = np.frombuffer(f.read(), np.uint8, offset=16)
        mnist = mnist.reshape(-1, 28, 28)

    datadir = '/home/acomasma/FactorDynamics/DMD-Mov-MNIST'
    ind = np.load(os.path.join(datadir, 'mnist_shuffle_ids.npy'))
    mnist = mnist[ind]
    return mnist

def load_fixed_set(root, is_train):
    # Load the fixed dataset
    step_length = .15
    alpha = 55
    crop_size = 64
    image_size = 64
    num_missing = 0
    n_f_in = 10
    n_f_out = 1
    datadir = '/home/acomasma/FactorDynamics/DMD-Mov-MNIST'
    name = 'vis_perc_{}-Nmiss_{}-100xStep_{}-alpha_{}-in{}_out{}-split-80-20-test.npz' \
        .format(int((crop_size / image_size) * 100),
                num_missing, int(step_length * 100), alpha,
                n_f_in, n_f_out)
    path = os.path.join(datadir, name)
    print(path)
    dataset = np.load(path)
    dataset = dataset.f.arr_0[..., 0, :, :, np.newaxis]
    # print(dataset.sum())

    return dataset

def compute_accelerating_theta(theta_0, theta_length, z_offset, x_offset,av_x, av_y):
    del_theta = np.pi*2/theta_length
    theta =theta_0
    x_array =np.array([])
    y_array =np.array([])
    z_array =np.array([])
    for i in range(theta_length):
        z = np.cos(2 * theta + z_offset)  # prev 1*theta
        r = np.sqrt(1 - z ** 2)
        x = r * np.sin(av_x * theta + x_offset)
        y = r * np.cos(av_y * theta)
        d_theta = 1+0.5*z
        theta =np.clip(theta+d_theta*del_theta, a_min=-np.pi, a_max=3*np.pi)
        x_array = np.append(x_array, x)
        y_array = np.append(y_array, y)
        z_array = np.append(z_array, z)
    return x_array, y_array, z_array

class MovingMNISTDataset(data.Dataset):
    def __init__(self, case, root, train, n_frames_input, n_frames_output, num_objects,
                 image_size = 128, train_split=0.8, transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNISTDataset, self).__init__()

        self.motion_type = case
        self.elast_c = 0.8 # was 1.1 without inversions
        self.visible_size = 35 #int(0.5 * image_size) # was 1.1 without inversions
        self.occlusion_num = 0

        self.image_size_ = image_size
        self.step_length_ = 0.3 # was 0.25

        self.dataset = None
        if train:
            self.mnist = load_mnist(root)
            self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]
        else:
            self.mnist = load_mnist(root)
            # self.dataset = load_fixed_set(root, False)
            self.length = int(2000) if self.dataset is None else self.dataset.shape[1]
            # TODO: if "generate", generate the dataset with the missing index.

        idx_full = np.arange(self.length)
        np.random.seed(0)
        np.random.shuffle(idx_full)
        self.split = idx_full

        self.is_train = train
        self.digit_size_ = 28
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform


    def get_random_trajectory(self, seq_length, motion_type='constant_vel'):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size_ - self.digit_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)
        # v_x = 0

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        if  motion_type=='circles' or motion_type=='circles_crop' or motion_type=='circles_missing':
            y = y * 0.6 + 0.2
            x = x * 0.6 + 0.2

            R = random.random()*0.6 + 0.2
            if x + R >= 1.0:
                R = 1 - random.random()*0.1 - x
            if x - R <= 0:
                R = x - random.random()*0.1
            if y + R >= 1.0:
                R = 1 - random.random()*0.1 - y
            if y - R <= 0:
                R = y - random.random()*0.1

            factor = 2
            self.step_length_ = 0.2 + random.random() * 0.1
            t = np.linspace(0, seq_length * self.step_length_ * factor, seq_length)

            if random.random() < 0.5:
                t = np.flip(t, axis=0)

            start_x, start_y = R*np.cos(t) + x, R*np.sin(t) + y

        if motion_type == 'constant_vel':
            for i in range(seq_length):
                # Take a step along velocity.
                y += v_y * self.step_length_
                x += v_x * self.step_length_

                # Bounce off edges.
                if x <= 0:
                    x = 0
                    v_x = -v_x * self.elast_c
                if x >= 1.0:
                    x = 1.0
                    v_x = -v_x * (1/self.elast_c)
                if y <= 0:
                    y = 0
                    v_y = -v_y * self.elast_c
                if y >= 1.0:
                    y = 1.0
                    v_y = -v_y * (1/self.elast_c)
                start_y[i] = y
                start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total, motion_type=self.motion_type)

            len_mnist = self.mnist.shape[0]
            train_len = int(.8*len_mnist) # Note: 0.8 is hardcoded but it's good enough.
            if self.is_train:
                ind = random.randint(0, train_len-1)
            else:
                ind = random.randint(train_len, len_mnist-1)
            digit_image = self.mnist[ind]


            occ_ids = []
            # if self.motion_type is 'circles_missing':
            #     # occ_frame_ids = range(self.n_frames_input - self.occlusion_num + 1)
            #     occ_frame_ids = range(self.n_frames_input - self.occlusion_num - 3)
            #     occ_id = random.sample(occ_frame_ids, 1)
            #     occ_ids = range(occ_id[0] + 2, occ_id[0] + self.occlusion_num + 2)  # It can't be frame in t=0

            for i in range(self.n_frames_total):
                top    = start_y[i]
                left   = start_x[i]
                bottom = top + self.digit_size_
                right  = left + self.digit_size_
                # Draw digit

                if not self.motion_type is 'circles_missing' or any(np.equal(i, occ_ids)):
                    data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train or self.num_objects != 2:
            # Sample number of objects
            num_digits = self.num_objects
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            num_digits = self.num_objects
            images = self.generate_moving_mnist(num_digits)
            # images = self.dataset[:, idx, ...]
        if self.motion_type == 'circles_crop':
            images = np.stack([self.crop_top_left_keepdim(images[i, ..., 0], self.visible_size, None)
                          for i in range(images.shape[0])])[..., np.newaxis]
            # print(images.shape)
        # RGB
        images = np.repeat(images, 3, axis=-1)
        if self.transform is not None:
            images = self.transform(images)
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        return input #, output

    def __len__(self):
        length = self.length
        return length

    def crop_top_left_keepdim(self, img, cropx, cropy):
        y, x = img.shape
        # img[:, :(x - cropx)] = 0
        img[:(y - cropx), :] = 0
        return img

class MovingMNISTDataset_centered(data.Dataset):
    def __init__(self, case, root, train, n_frames_input, n_frames_output, num_objects,
                 image_size = 128, train_split=0.8, transform=None):
        '''
        param num_objects: a list of number of possible objects.
        '''
        super(MovingMNISTDataset_centered, self).__init__()

        self.motion_type = case

        self.image_size_ = image_size
        self.digit_size_ = 28
        self.max_digit_size = 28
        self.step_length_ = 0.25

        self.dataset = None
        if train:
            self.mnist = load_mnist(root)
            self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]
        else:
            self.mnist = load_mnist(root)
            # self.dataset = load_fixed_set(root, False)
            self.length = int(2000) if self.dataset is None else self.dataset.shape[1]

        # idx_full = np.arange(self.length)
        # np.random.seed(0)
        # np.random.shuffle(idx_full)
        # self.split = idx_full

        self.is_train = train
        # self.digit_size_ = 20
        self.num_objects = num_objects
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform

    def get_random_trajectory(self, seq_length, motion_type='constant_vel'):
        ''' Generate a random sequence of a MNIST digit '''
        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)

        if motion_type == 'spherical_manifold':
            # author Sandesh Ghimire (contact: drsandeshghimire@gmail.com)
            av_x = 6  # 3
            av_y = 6  # 3
            theta_length = 100
            z_offset = 3 * np.pi / 4 + (3 * np.pi / 4) * np.random.rand()  # prev pi/3
            x_offset = np.pi / 6 + (np.pi / 4) * np.random.rand()

            theta = np.linspace(0, 2 * np.pi, theta_length)
            z = np.cos(2 * theta + z_offset)  # prev 1*theta
            r = np.sqrt(1 - z ** 2)
            x = r * np.sin(av_x * theta + x_offset)
            y = r * np.cos(av_y * theta)
            # Rotate
            rot = Ro.from_rotvec([np.pi / 8, np.pi / 8, np.pi / 8])
            R_t = rot.as_matrix()
            p = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)),
                               axis=0)
            random_seq_chunk = generate_random_chunk(theta_length, seq_length)
            p_new = np.matmul(R_t, p[:, random_seq_chunk])

            # plot_3Dcurve(p_new[0], p_new[1], p_new[2], 'random_mnist_curve')
            # porjection
            screen_pos = -1
            focal_pos = -3
            # Idea: object moves inside a cube from [-1,1]^3 Screen is positioned at z =-1 and focal point  is at z = -1.5
            projection_factor = abs(screen_pos - focal_pos) / (p_new[2] - focal_pos)  # (s-f)/(z-f)
            self.digit_resize_factor = projection_factor
            max_factor =max(self.digit_resize_factor)
            self.max_digit_size = max(max_factor*self.digit_size_, self.max_digit_size )

            x_proj = p_new[0] * projection_factor
            y_proj = p_new[1] * projection_factor

            # moving x and y from [-1, +1] to [0,1]
            x_unitized, y_unitized = 0.5 * (x_proj + 1), 0.5 * (
                    y_proj + 1)  # By design this should never be outside [0,1]

            max_val = max(max(x_unitized), max(y_unitized))
            min_val = min(min(x_unitized), min(y_unitized))
            # This causes shift and stretching. I think that would introduce additional nonlinearity. But, I had to do
            # it to make the digits take the whole space in frame. My first experiment doesn't do this step.
            start_y = (y_unitized - min_val) / (max_val - min_val)
            start_x = (x_unitized - min_val) / (max_val - min_val)

            # plot_2Dcurve(start_x, start_y, '2D_random_mnist_curve')
            # print('plot_on')

        if motion_type == 'spherical_manifold_accelerate':
            # author Sandesh Ghimire (contact: drsandeshghimire@gmail.com)
            av_x = 6  # 3
            av_y = 6  # 3
            theta_length = 100
            z_offset = 3 * np.pi / 4 + (3 * np.pi / 4) * np.random.rand()  # prev pi/3
            x_offset = np.pi / 6 + (np.pi / 4) * np.random.rand()

            x, y, z = compute_accelerating_theta(0, theta_length, z_offset, x_offset,av_x, av_y)

            # theta = np.linspace(0, 2 * np.pi, theta_length)
            # Rotate
            rot = Ro.from_rotvec([np.pi / 8, np.pi / 8, np.pi / 8])
            R_t = rot.as_matrix()
            p = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), np.expand_dims(z, axis=0)),axis=0)
            random_seq_chunk = generate_random_chunk(theta_length, seq_length)
            p_new = np.matmul(R_t, p[:, random_seq_chunk])

            # plot_3Dcurve(p_new[0], p_new[1], p_new[2], 'random_mnist_curve')
            # porjection
            screen_pos = -1
            focal_pos = -3
            # Idea: object moves inside a cube from [-1,1]^3 Screen is positioned at z =-1 and focal point  is at z = -1.5
            projection_factor = abs(screen_pos - focal_pos) / (p_new[2] - focal_pos)  # (s-f)/(z-f)
            self.digit_resize_factor = projection_factor
            max_factor = max(self.digit_resize_factor)
            self.max_digit_size = max(max_factor * self.digit_size_, self.max_digit_size)

            x_proj = p_new[0] * projection_factor
            y_proj = p_new[1] * projection_factor

            # moving x and y from [-1, +1] to [0,1]
            x_unitized, y_unitized = 0.5 * (x_proj + 1), 0.5 * (
                    y_proj + 1)  # By design this should never be outside [0,1]

            max_val = max(max(x_unitized), max(y_unitized))
            min_val = min(min(x_unitized), min(y_unitized))
            # This causes shift and stretching. I think that would introduce additional nonlinearity. But, I had to do
            # it to make the digits take the whole space in frame. My first experiment doesn't do this step.
            start_y = (y_unitized - min_val) / (max_val - min_val)
            start_x = (x_unitized - min_val) / (max_val - min_val)

            # plot_2Dcurve(start_x, start_y, '2D_random_mnist_curve')
            # print('plot_on')

            # Scale to the size of the canvas.
        # Scale to the size of the canvas.
        canvas_size = self.image_size_ - np.ceil(self.max_digit_size)
        start_y = (canvas_size * start_y + self.max_digit_size/2).astype(np.int32)
        start_x = (canvas_size * start_x + self.max_digit_size/2).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self, num_digits=2):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.n_frames_total, self.image_size_, self.image_size_), dtype=np.float32)
        for n in range(num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.n_frames_total, motion_type=self.motion_type)

            len_mnist = self.mnist.shape[0]
            train_len = int(.8*len_mnist) # Note: 0.8 is hardcoded but it's good enough.
            if self.is_train:
                ind = random.randint(0, train_len-1)
            else:
                ind = random.randint(train_len, len_mnist-1)
            random_image = self.mnist[ind]

            for i in range(self.n_frames_total):
                width= (self.digit_size_*self.digit_resize_factor[i]).astype(np.int32)
                height = (self.digit_size_ * self.digit_resize_factor[i]).astype(np.int32)
                digit_image = cv2.resize(random_image, (width, height), interpolation = cv2.INTER_CUBIC)
                # digit_size = digit_image.shape[0]
                top = (start_y[i] - height/2).astype(np.int32)
                left = (start_x[i] - width/2).astype(np.int32)
                bottom = (top+height).astype(np.int32)
                right = (left + height).astype(np.int32)
                # Draw digit
                try:
                    data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)
                except(ValueError):
                    print('size mismatch')

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output
        if self.is_train:
            # Sample number of objects
            num_digits = self.num_objects
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)
        else:
            num_digits = self.num_objects
            # Generate data on the fly
            images = self.generate_moving_mnist(num_digits)

        if self.transform is not None:
            images = self.transform(images)
        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        return input# , output

    def __len__(self):
        return self.length