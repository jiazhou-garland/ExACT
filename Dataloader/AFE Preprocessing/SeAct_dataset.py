from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2, os, json
from dv import AedatFile

class DVSReader(object):
    def __init__(self, fileName):
        super(DVSReader, self).__init__()
        self.C = 3

        with AedatFile(fileName) as f:
            self.height, self.width = f['events'].size
            self.Events = np.hstack([packet for packet in f['events'].numpy()])
            self.tE = self.Events['timestamp']
            self.xE = self.Events['x']
            self.yE = self.Events['y']
            self.pE = 2 * self.Events['polarity'] - 1

            tImg = []
            Img = []
            # Img H*W*1
            for packet in f['frames']:
                tImg.append(packet.timestamp)
                Img.append(packet.image)
            self.tImg = tImg
            self.Img = Img

class SeACT(Dataset):
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
        self.sample_event_num_min = 100000
        self.sample_event_threshold = 40
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
        # label
        label_str = event_stream_path.split('/')[7].split('-')[0][-4:]
        label_idx = int(label_str)
        print(label_idx)

        event = DVSReader(event_stream_path)
        ts = event.tE.tolist()
        x = event.xE.tolist()
        y = event.yE.tolist()
        pol = event.pE.tolist()
        events_stream = np.array([x,y,ts,pol]).transpose()

        # N, _ = events_stream.shape
        # print("the event number is:" + str(N))

        all_frame = self.adaptive_event_sampling(events_stream)
        all_frame = np.array(all_frame)
        events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W

        return events_data, label_idx, event_stream_path

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        # random.shuffle(self.files)
        return len(self.files)

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
        np.add.at(img_neg, x[p == -1] + w_event * y[p == -1], 1)

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
            event_frame = (np.clip(img_pos, 0, 1) + np.clip(img_neg, 0, 1)) * [127.5,127.5,127.5]

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

        w_event = 261
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
            # print('N:'+ str(N))
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
        # print(N)
        half_N = int(np.floor(N / 2))
        # print(half_N)
        half_frame1 = self.generate_abs_image(events_stream[:half_N, :])
        # print(half_frame1.sum())
        half_frame2 = self.generate_abs_image(events_stream[half_N:, :])
        diff_image = np.abs(half_frame1 - half_frame2)
        idx = 200 * (diff_image.sum() / N)
        print(idx)

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
    all_path = r'Path-to-/DVS Semantic Action_whole.txt' # TODO: Change to your directory
    num_events = 100000 # 1093470 546735 273367 136683 68341
    median_length = 1500000
    frame = 1513
    tf = open("Path-to-/DVS Semantic Action.json", "r") # TODO: Change to your directory
    classnames_dict = json.load(tf)  # class name idx start from 0
    key_list = list(classnames_dict.keys())
    val_list = list(classnames_dict.values())

    datasets = SeACT(all_path, representation='rgb', median_length = median_length,
                   num_events = num_events, frame=frame, augmentation=False)
    feeder = DataLoader(datasets, batch_size=1, shuffle=False)
    for step, (events_image, class_idxs, event_stream_path) in enumerate(feeder):
        ind = val_list.index(class_idxs.numpy()[0])
        print()
        events_image = events_image.numpy().transpose(0,2,3,4,1) # B,T,H,W,C
        B, T, H, W, C = events_image.shape
        for i in range(B):
            for j in range(T):
                file_path = event_stream_path[0].replace('DVS Semantic Action', 'DVS Semantic Action Sampled').replace('.aedat', '')
                folder_path = '/'.join([file_path.split('/')[i] for i in range(7)])
                # print(file_path)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                class_path = '/'.join([folder_path, file_path.split('/')[7].split('-')[0][-4:]])
                # print(class_path)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                file_path = class_path + '/' + str(j) +'.png'
                print(file_path)
                img = events_image[i,j,:,:,:]
                cv2.imwrite(file_path, img)




