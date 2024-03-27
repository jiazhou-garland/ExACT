from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import struct
import numpy as np
import cv2,os

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


        all_frame = self.adaptive_event_sampling(events_stream)
        all_frame = np.array(all_frame)
        # print(all_frame.shape)
        events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W

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
        if representation == 'rgb':
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
            return [current_frame]

        if divide_N <= self.sample_event_num_min: # return the event frame if the event number is smaller than the default minimum number.
            half_frame1 = self.generate_event_image(events_stream[:divide_N, :], (self.height, self.width), self.representation)
            half_frame2 = self.generate_event_image(events_stream[divide_N:, :], (self.height, self.width), self.representation)
            return [half_frame1, half_frame2]

        # For unsufficiently sampled and divide_N > self.sample_event_num_min,
        # evenly divide the event stream and then sampled the two divided event streams recursively.
        frame_list1 = self.adaptive_event_sampling(events_stream[:divide_N, :])
        frame_list2 = self.adaptive_event_sampling(events_stream[divide_N:, :])
        return np.concatenate((frame_list1, frame_list2),axis=0)

    def if_sufficiently_sampled(self, events_stream):
        N, _ = events_stream.shape
        half_N = int(np.floor(N / 2))
        half_frame1 = self.generate_abs_image(events_stream[:half_N, :])
        half_frame2 = self.generate_abs_image(events_stream[half_N:, :])
        diff_image = np.abs(half_frame1 - half_frame2)
        idx = 200 * (diff_image.sum() / N)

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
        h2, w2 = img.shape[0:2]
        scale = self.height * 1.0 / h2
        scale2 = self.width * 1.0 / w2
        img = cv2.resize(img, dsize=None, fx=scale2, fy=scale)
        return img




if __name__ == '__main__':
    all_path = r'Path-to-/HARDVS_whole.txt' # TODO: Change to your directory
    num_events = 80000 # 1093470 546735 273367 136683 68341
    median_length = 1500000
    frame = 1513
    tf = open("Path-to-/HARDVS_300_class.json", "r")  # TODO: Change to your directory
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
                file_path = event_stream_path[0].replace('HARDVS', 'HARDVS_Sampled')
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


