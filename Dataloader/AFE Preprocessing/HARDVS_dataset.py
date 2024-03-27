from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import struct
import copy, json
import numpy as np
import matplotlib.pyplot as plt
import cv2,os
import pickle
# from spikingjelly.datasets.cifar10_dvs import load_events

def random_shift_events(events, max_shift=20, resolution=(180, 240)):
    H, W = resolution
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
    events[:, 0] += x_shift
    events[:, 1] += y_shift

    valid_events = (events[:, 0] >= 0) & (events[:, 0] < W) & (events[:, 1] >= 0) & (events[:, 1] < H)
    events = events[valid_events]

    return events

def random_flip_events_along_x(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    flag = 0
    if np.random.random() < p:
        events[:, 0] = W - 1 - events[:, 0]
        flag = 1
    return events, flag

def random_flip_events_along_y(events, resolution=(180, 240), p=0.5):
    H, W = resolution
    flag = 0
    if np.random.random() < p:
        events[:, 1] = H - 1 - events[:, 1]
        flag = 1
    return events, flag

class HARDVS(Dataset):
    def __init__(self, txtPath, num_events=100000, median_length=100000,
                 frame=6, resize_width=224,
                 resize_height=224, representation=None,
                 augmentation=False, pad_frame_255=False):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        self.augmentation = augmentation
        self.width, self.height = resize_width, resize_height
        self.representation = representation
        self.frame = frame
        self.num_events = num_events
        self.median_length = median_length
        self.pad_frame_255 = pad_frame_255
        tf = open('/home/jinjing/zhoujiazhou/HumanECLIP/Dataloader/HARDVS/HARDVS_300_class.json', "r")
        self.classnames_dict = json.load(tf)  # class name idx start from 0
        self.sample_event_num_min = 150000
        self.sample_event_threshold = 0
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path = self.files[idx].split('\t')[0][:-1]
        # print(event_stream_path)
        # label
        label_idx = int(event_stream_path.split('/')[7][7:])
        # print(label_idx)

        events_stream = np.load(event_stream_path)
        x, y, ts, pol = events_stream['x'], events_stream['y'], events_stream['t'], events_stream['p']
        events_stream = np.array([x,y,ts,pol]).transpose()
        # print(events_stream.shape)

        # if self.augmentation:
        #     events_stream = random_flip_events_along_x(events_stream, (self.height, self.width))

        # real_n, _  = events_stream.shape
        # real_num_frame = int(real_n / self.num_events)
        # events_stream, pad_flag = self.pad_event_stream(events_stream, median_length=self.median_length)
        N, _ = events_stream.shape
        # num_frame = int(np.floor( N / self.num_events))
        # print(N)
        # all_frame = []
        # if num_frame == 0:
        #     events_image = self.generate_event_image(events_stream, (self.height, self.width), self.representation)
        #     all_frame.append(events_image)
        # for i in range(num_frame):
        #     # if pad_flag and i > real_num_frame and self.pad_frame_255:
        #     #     all_frame.append(255*np.ones((self.height, self.width, 3), dtype=np.float))
        #     # else:
        #     events_tmp = events_stream[i * self.num_events: (i + 1) * self.num_events, :]
        #     events_image = self.generate_event_image(events_tmp, (self.height, self.width), self.representation)
        #     events_image = cv2.flip(events_image, 0)
        #     all_frame.append(events_image)
        #
        #     N, _ = events_tmp.shape
        #     # print(N)
        #     half_N = int(np.floor(N / 2))
        #     # print(half_N)
        #     half_frame1 = self.generate_event_image(events_tmp[:half_N,:], (self.height, self.width), self.representation)
        #     half_frame1 = cv2.flip(half_frame1, 0)
        #     half_frame1_clip = 1000*np.clip(half_frame1,0,0.001)
        #     # print(half_frame1.sum())
        #     half_frame2 = self.generate_event_image(events_tmp[half_N:,:], (self.height, self.width), self.representation)
        #     half_frame2 = cv2.flip(half_frame2, 0)
        #     half_frame2_clip = 1000*np.clip(half_frame2,0,0.001)
        #     # print(half_frame2.sum())
        #
        #     # kernel_dilate = np.ones((3, 3), np.uint8)
        #     # kernel_erode = np.ones((3, 3), np.uint8)
        #     # half_frame1 = cv2.erode(half_frame1, kernel_erode, iterations=3)
        #     # half_frame1 = cv2.dilate(half_frame1, kernel_dilate, iterations=3)
        #     # half_frame2 = cv2.erode(half_frame2, kernel_erode, iterations=3)
        #     # half_frame2 = cv2.dilate(half_frame2, kernel_dilate, iterations=3)
        #     # print(half_frame1_clip.sum())
        #     # print(half_frame2_clip.sum())
        #     diff_image = np.abs(half_frame1_clip - half_frame2_clip)
        #     # print(1000*np.clip(half_frame2,0,0.001))
        #     # print(np.abs(diff_image).sum())
        #     print(200 *(np.abs(diff_image).sum() / (half_frame1_clip.sum()+half_frame2_clip.sum())))
        #     # all_frame.append(half_frame1)
        #     # all_frame.append(half_frame2)
        #     all_frame.append(diff_image)
        #
        # events_data = np.array(all_frame).transpose(3, 0, 1, 2)

        all_frame = self.adaptive_event_sampling(events_stream)
        all_frame = np.array(all_frame)
        # print(all_frame.shape)
        events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W

        # elif self.representation == 'mlp_learned':
        #     events_data,_ = self.pad_event_stream(events_stream)
        #     # print(events_data)

        return events_data, label_idx, event_stream_path

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        # random.shuffle(self.files)
        return len(self.files)

    def getDVSeventsDavis(self, file, numEvents=1e10, startTime=0):
        """ DESCRIPTION: This function reads a given aedat file and converts it into four lists indicating
                         timestamps, x-coordinates, y-coordinates and polarities of the event stream.

        Args:
            file: the path of the file to be read, including extension (str).
            numEvents: the maximum number of events allowed to be read (int, default value=1e10).
            startTime: the start event timestamp (in microseconds) where the conversion process begins (int, default value=0).

        Return:
            ts: list of timestamps in microseconds.
            x: list of x-coordinates in pixels.
            y: list of y-coordinates in pixels.
            pol: list of polarities (0: on -> off, 1: off -> on).
        """
        # print('\ngetDVSeventsDavis function called \n')
        sizeX = 346
        sizeY = 260
        x0 = 0
        y0 = 0
        x1 = sizeX
        y1 = sizeY

        # print('Reading in at most', str(numEvents))

        triggerevent = int('400', 16)
        polmask = int('800', 16)
        xmask = int('003FF000', 16)
        ymask = int('7FC00000', 16)
        typemask = int('80000000', 16)
        typedvs = int('00', 16)
        xshift = 12
        yshift = 22
        polshift = 11
        x = []
        y = []
        ts = []
        pol = []
        numeventsread = 0

        length = 0
        aerdatafh = open(file, 'rb')
        k = 0
        p = 0
        statinfo = os.stat(file)
        if length == 0:
            length = statinfo.st_size
        # print("file size", length)

        lt = aerdatafh.readline()
        while lt and str(lt)[2] == "#":
            p += len(lt)
            k += 1
            lt = aerdatafh.readline()
            continue

        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        while p < length:
            ad, tm = struct.unpack_from('>II', tmp)
            ad = abs(ad)
            if tm >= startTime:
                if (ad & typemask) == typedvs:
                    xo = sizeX - 1 - float((ad & xmask) >> xshift)
                    yo = float((ad & ymask) >> yshift)
                    polo = 1 - float((ad & polmask) >> polshift)
                    if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                        x.append(xo)
                        y.append(yo)
                        pol.append(polo)
                        ts.append(tm)
            aerdatafh.seek(p)
            tmp = aerdatafh.read(8)
            p += 8
            numeventsread += 1

        print('Total number of events read =', numeventsread)
        # print('Total number of DVS events returned =', len(ts))
        return ts, x, y, pol

    def pad_event_stream(self, event_stream, median_length = 104815):
        """
        pad event stream along n dim with 0
        so that event streams in one batch have the same dimension
        """
        # max_length = 428595
        pad_flag = False
        (N, _) = event_stream.shape
        if N < median_length:
            n = median_length - N
            pad = np.ones((n, 4))
            event_stream = np.concatenate((event_stream, pad), axis=0)
            pad_flag = True
        else:
            event_stream = event_stream[:median_length, :]
        return event_stream, pad_flag

    def generate_event_image(self, events_clip, shape, representation):
        """
        events_clip: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}.
        x and y correspond to image coordinates u and v.
        """
        H, W = shape
        x, y, t, p = events_clip.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = x.max() + 1
        h_event = y.max() + 1
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)

        img_pos = img_pos.reshape((h_event, w_event,1))
        img_neg = img_neg.reshape((h_event, w_event,1))

        # denoising using morphological transformation
        kernel_dilate = np.ones((2,2), np.uint8)
        kernel_erode = np.ones((2,2), np.uint8)
        img_pos = cv2.erode(img_pos, kernel_erode, iterations=1)
        img_neg = cv2.erode(img_neg, kernel_erode, iterations=1)
        img_pos = cv2.dilate(img_pos, kernel_dilate, iterations=1)
        img_neg = cv2.dilate(img_neg, kernel_dilate, iterations=1)

        img_pos = img_pos.reshape((h_event, w_event,1))
        img_neg = img_neg.reshape((h_event, w_event,1))
        # img_pos = img_pos / (np.max(img_pos) + 1e-12)
        # img_neg = img_neg / (np.max(img_neg) + 1e-12)

        if representation == 'rgb':
            # event_frame = (img_pos * [0, 0, 255] + img_neg * [255,0,0])
            event_frame = 255*(1 - (img_pos.reshape((h_event, w_event, 1))* [0, 255, 255] + img_neg.reshape((h_event, w_event, 1)) * [255,255,0]) / 255)
        elif representation == 'gray_scale':
            event_frame = (img_pos + img_neg) * [127,127,127]

        event_frame = np.clip(event_frame, 0, 255)
        # scale
        scale = H * 1.0 / h_event
        scale2 = W * 1.0 / w_event
        event_frame = cv2.resize(event_frame, dsize=None, fx=scale2, fy=scale)

        return event_frame

    def generate_abs_image(self, events_clip):
        """
        generate event image without normalization, resize
        """
        x, y, t, p = events_clip.T
        x = x.astype(np.int32)
        y = y.astype(np.int32)

        w_event = 260
        h_event = 346
        img_pos = np.zeros((h_event * w_event,), dtype="float32")
        img_neg = np.zeros((h_event * w_event,), dtype="float32")
        np.add.at(img_pos, x[p == 1] + w_event * y[p == 1], 1)
        np.add.at(img_neg, x[p == 0] + w_event * y[p == 0], 1)

        img_pos = img_pos.reshape((h_event, w_event,1))
        img_neg = img_neg.reshape((h_event, w_event,1))

        # denoising using morphological transformation
        kernel_dilate = np.ones((2,2), np.uint8)
        kernel_erode = np.ones((2,2), np.uint8)
        img_pos = cv2.erode(img_pos, kernel_erode, iterations=1)
        img_neg = cv2.erode(img_neg, kernel_erode, iterations=1)
        img_pos = cv2.dilate(img_pos, kernel_dilate, iterations=1)
        img_neg = cv2.dilate(img_neg, kernel_dilate, iterations=1)

        img_pos = img_pos.reshape((h_event, w_event,1))
        img_neg = img_neg.reshape((h_event, w_event,1))

        gray_scale = (img_pos+img_neg) * [1,1,1]

        return gray_scale

    def adaptive_event_sampling(self, events_stream):

        N, _  = events_stream.shape
        divide_N = int(np.floor(N / 2))

        if self.if_sufficiently_sampled(events_stream): # return True for sufficiently sampled, no need for proceed.
            current_frame = self.generate_event_image(events_stream, (self.height, self.width), self.representation)
            # current_frame = cv2.flip(current_frame, 0)
            # print('N:'+ str(N))
            return [current_frame]

        if divide_N <= self.sample_event_num_min: # return the event frame if the event number is smaller than the default minimum number.
            half_frame1 = self.generate_event_image(events_stream[:divide_N, :], (self.height, self.width), self.representation)
            # half_frame1 = cv2.flip(half_frame1, 0)
            half_frame2 = self.generate_event_image(events_stream[divide_N:, :], (self.height, self.width), self.representation)
            # half_frame2 = cv2.flip(half_frame2, 0)
            return [half_frame1, half_frame2]

        # For unsufficiently sampled and divide_N > self.sample_event_num_min,
        # evenly divide the event stream and then sampled the two divided event streams recursively.
        frame_list1 = self.adaptive_event_sampling(events_stream[:divide_N, :])
        frame_list2 = self.adaptive_event_sampling(events_stream[divide_N:, :])
        return np.concatenate((frame_list1, frame_list2),axis=0)

    def if_sufficiently_sampled(self, events_stream):
        N, _ = events_stream.shape
        # print(N)
        half_N = int(np.floor(N / 2))
        # print(half_N)
        half_frame1 = self.generate_abs_image(events_stream[:half_N, :])
        # print(half_frame1.sum())
        half_frame2 = self.generate_abs_image(events_stream[half_N:, :])
        diff_image = np.abs(half_frame1 - half_frame2)
        idx = 200 * (diff_image.sum() / N)
        # print(idx)
        # print(diff_image.sum())
        # print(half_frame1.sum())
        # print(half_frame2.sum())

        # plt.figure()
        # plt.imshow(diff_image)
        # plt.axis('off')
        # plt.show()

        if idx <= self.sample_event_threshold:
            return True
        else:
            return False

    def scale_image(self, img):
        """
        0.For binary image, transform it into gray image by letting img=R=G=B
        1.Pad the image lower than H,W,3 with 255
        2.Resize the padded image to H,W,3
        """
        # for binary image
        if img.ndim == 2:
            img = np.array([img, img, img]).transpose(1, 2, 0)  # H,W,3

        # h, w, _ = img.shape
        # a = self.height - h
        # b = self.width - w
        #
        # if a > 0:
        #     img = np.pad(img, ((0, a), (0, 0), (0, 0)), "constant", constant_values=255)
        # if b > 0:
        #     img = np.pad(img, ((0, 0), (0, b), (0, 0)), "constant", constant_values=255)

        h2, w2 = img.shape[0:2]
        scale = self.height * 1.0 / h2
        scale2 = self.width * 1.0 / w2
        img = cv2.resize(img, dsize=None, fx=scale2, fy=scale)
        return img

def visualize_events_stream(events):
    p = copy.deepcopy(events[:, 3])
    events_neg = events[p == 0, :]
    events_pos = events[p == 1, :]

    # Creating figures for the plot
    # fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")
    # Creating a plot using the random datasets
    ax.scatter3D(events_pos[:, 0], events_pos[:, 1], events_pos[:, 2], color="red", marker='')
    ax.scatter3D(events_neg[:, 0], events_neg[:, 1], events_neg[:, 2], color="blue", marker='')
    ax.set_xlabel('X-axis', fontweight='bold')
    ax.set_ylabel('Y-axis', fontweight='bold')
    ax.set_zlabel('T-axis', fontweight='bold')
    plt.title("Event stream plot")
    # display the plot
    plt.show()

def visualize_events_image(events, frame):
    """Visualizes the input histogram
    events:[Batch, N, 4]
    """
    events = events[0, :, :]
    x, y, t, p = events.T
    N, _ = events.shape
    time_window = int(N / frame)
    for i in range(frame):
        x_ = x[i * time_window: (i + 1) * time_window]
        y_ = y[i * time_window: (i + 1) * time_window]
        p_ = p[i * time_window: (i + 1) * time_window]

        plt.figure()
        plt.scatter(x_[p_ == 1], y_[p_ == 1], color="red", marker='.')

        plt.scatter(x_[p_ == 0], y_[p_ == 0], color="blue", marker='.')
        plt.axis('off')
        plt.show()

def visualize_whole_data_stream(event_count, image, label, frame):
    B = event_count.shape[0]
    plt_num = int((frame + 1) / 3 + 1)
    for j in range(B):
        plt.figure()
        for i in range(frame):
            histogram = event_count[j, i, :, :, :]
            height, width, _ = histogram.shape
            np_image = np.zeros([height, width, 3])

            # H,W,1 * 1,1,3 ->H,W,3
            # histogram[:, :, 1]->pos red->np.array([1, 0, 0]
            np_image += (histogram[:, :, 1])[:, :, None] * np.array([1, 0, 0])[None, None, :]
            # H,W,1 * 1,1,3 ->H,W,3
            # histogram[:, :, 0]->neg blue->np.array([0, 0, 1]
            np_image += (histogram[:, :, 0])[:, :, None] * np.array([0, 0.5, 0.4])[None, None, :]
            np_image = np_image.clip(0, 1)
            # np_image[np.where(np_image == 0)] = 1

            plt.subplot(3, plt_num, i + 1)
            plt.imshow(np_image.astype(np.float64))
            plt.title('frame: ' + str(i + 1))
            plt.axis('off')

        plt.subplot(3, plt_num, frame + 1)
        plt.imshow(image[j, :, :, :])
        plt.axis('off')
        plt.title(label[j])
        plt.show()

def visualize_grayscale_img(grayscale_img):
    # print(image.shape)
    B,T,H,W,C = grayscale_img.shape
    plt_num = int((T + 1) / 3 + 1)
    for j in range(B):
        plt.figure()
        for i in range(T):
            img = grayscale_img[j, i, :, :, :]
            plt.subplot(3, plt_num, i + 1)
            plt.imshow(img)
            plt.axis('off')

        plt.show()

def analysis_dataset(path):
    length_t = []
    with open(path, 'r') as f:
        for line in f.readlines():
            # event
            event_stream_path, image_path = line.split('\t')
            raw_data = np.fromfile(open(event_stream_path, 'rb'), dtype=np.uint8)
            raw_data = np.int32(raw_data)
            # all_y = raw_data[1::5]
            # all_x = raw_data[0::5]
            # all_p = (raw_data[2::5] & 128) >> 7  # bit 7
            all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
            length_t.append(len(all_ts))
    length_t = np.array(length_t)
    print('max: ' + str(length_t.max()) + ' min: ' + str(length_t.min()))


if __name__ == '__main__':
    train_path = r"/zjz/HumanECLIP/Dataloader/PAF/PAF_train.txt"
    # all_path = r'/zjz/HumanECLIP/Dataloader/PAF/PAF_whole.txt'
    all_path = r'/home/jinjing/zhoujiazhou/HumanECLIP/Dataloader/HARDVS/HARDVS_whole-1.txt'
    num_events = 80000 # 1093470 546735 273367 136683 68341
    median_length = 1500000
    frame = 1513
    tf = open("/home/jinjing/zhoujiazhou/HumanECLIP/Dataloader/HARDVS/HARDVS_300_class.json", "r")
    # tf = open("/zjz/HumanECLIP/Dataloader/PAF/PAF.json", "r")
    # classnames_dict = json.load(tf)  # class name idx start from 0
    # classnames_list = [i for i in classnames_dict.keys()]
    datasets = HARDVS(all_path, representation='rgb', median_length = median_length,
                   num_events = num_events, frame=frame, augmentation=False)
    feeder = DataLoader(datasets, batch_size=1, shuffle=False)
    file1 = []
    for step, (events_image, class_idxs, event_stream_path) in enumerate(feeder):
        # classnames = classnames_list[class_idxs]
        events_image = events_image.numpy().transpose(0,2,3,4,1) # B,T,H,W,C
        B, T, H, W, C = events_image.shape
        for i in range(B):
            for j in range(T):
                file_path = event_stream_path[0].replace('HARDVS', 'HARDVS_Sampledt35')
                folder_path = '/'.join([file_path.split('/')[i] for i in range(6)])
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                class_path = '/'.join([folder_path, file_path.split('/')[7]])
                # print(class_path)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                class_path2 = '/'.join([class_path, file_path.split('/')[8]])
                # print(class_path)
                if not os.path.exists(class_path2):
                    os.makedirs(class_path2)

                file_path = class_path2 + '/' + str(j) +'.png'
                img = events_image[i,j,:,:,:]
                cv2.imwrite(file_path, img)
        if T == 1:
            file1.append(file_path)
        print(f"step:{step+1}:" + file_path)

    with open('file1.txt', 'w') as f:
        for item in file1:
            f.write("%s\n" % item)

    # events_stream, class_idxs, event_stream_path = next(iter(feeder))
    # events_image = events_image.numpy().transpose(0,2,3,4,1) # B,T,H,W,C
    # visualize_grayscale_img(events_image)


    # visualize_events_stream(events_stream.squeeze(0))
    # visualize_events_image(events_stream, frame=frame)
    # print(events_image.shape)
    # analysis_dataset(train_path)

