from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import struct, json
import numpy as np
import cv2,os
class DVS128Gesture(Dataset):
    def __init__(self, txtPath, resize_width=224,
                 resize_height=224, representation=None,
                 augmentation=False, pad_frame_255=False):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        self.augmentation = augmentation
        self.width, self.height = resize_width, resize_height
        self.representation = representation
        self.pad_frame_255 = pad_frame_255
        self.sample_event_num_min = 15000
        self.sample_event_threshold = 30
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path = self.files[idx][:-1]
        events = self.load_aedat_v3(event_stream_path)

        cvs_path = event_stream_path.replace('.aedat','_labels.csv')
        csv_data = np.loadtxt(cvs_path, dtype=np.uint32, delimiter=',', skiprows=1)
        all_event_data = []
        all_label_idx = []
        for i in range(csv_data.shape[0]):
            # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
            label_idx = csv_data[i][0] - 1
            all_label_idx.append(label_idx)
            t_start = csv_data[i][1]
            t_end = csv_data[i][2]
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            t=events['t'][mask]
            x=events['x'][mask]
            y=events['y'][mask]
            p=events['p'][mask]
            events_stream = np.array([x, y, t, p]).transpose()
            all_frame = self.adaptive_event_sampling(events_stream)
            all_frame = np.array(all_frame)
            events_data = all_frame.transpose(3, 0, 1, 2)  # T,H,W,3 -> 3,T,H,W
            all_event_data.append(events_data)

        return all_event_data, all_label_idx, event_stream_path

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        # random.shuffle(self.files)
        return len(self.files)

    def load_aedat_v3(self, file_name: str):
        '''
        :param file_name: path of the aedat v3 file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict
        This function is written by referring to https://gitlab.com/inivation/dv/dv-python . It can be used for DVS128 Gesture.
        '''
        with open(file_name, 'rb') as bin_f:
            # skip ascii header
            line = bin_f.readline()
            while line.startswith(b'#'):
                if line == b'#!END-HEADER\r\n':
                    break
                else:
                    line = bin_f.readline()

            txyp = {
                't': [],
                'x': [],
                'y': [],
                'p': []
            }
            while True:
                header = bin_f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                e_type = struct.unpack('H', header[0:2])[0]
                e_source = struct.unpack('H', header[2:4])[0]
                e_size = struct.unpack('I', header[4:8])[0]
                e_offset = struct.unpack('I', header[8:12])[0]
                e_tsoverflow = struct.unpack('I', header[12:16])[0]
                e_capacity = struct.unpack('I', header[16:20])[0]
                e_number = struct.unpack('I', header[20:24])[0]
                e_valid = struct.unpack('I', header[24:28])[0]

                data_length = e_capacity * e_size
                data = bin_f.read(data_length)
                counter = 0

                if e_type == 1:
                    while data[counter:counter + e_size]:
                        aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                        timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                        x = (aer_data >> 17) & 0x00007FFF
                        y = (aer_data >> 2) & 0x00007FFF
                        pol = (aer_data >> 1) & 0x00000001
                        counter = counter + e_size
                        txyp['x'].append(x)
                        txyp['y'].append(y)
                        txyp['t'].append(timestamp)
                        txyp['p'].append(pol)
                else:
                    # non-polarity event packet, not implemented
                    pass
            txyp['x'] = np.asarray(txyp['x'])
            txyp['y'] = np.asarray(txyp['y'])
            txyp['t'] = np.asarray(txyp['t'])
            txyp['p'] = np.asarray(txyp['p'])
            return txyp

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

        w_event = 128
        h_event = 128
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
    train_path = r"Path-to-/DVS128Gesture_train.txt" # TODO: Change to your directory
    val_path = r"Path-to-/DVS128Gesture_val.txt" # TODO: Change to your directory
    tf = open("Path-to-/DVS128Gesture.json", "r") # TODO: Change to your directory
    classnames_dict = json.load(tf)  # class name idx start from 0
    classnames_list = [i for i in classnames_dict.keys()]
    # print(classnames_list)
    datasets = DVS128Gesture(val_path, representation='rgb', augmentation=False)
    feeder = DataLoader(datasets, batch_size=1, shuffle=False)
    for step, (events_image, class_idxs, event_stream_path) in enumerate(feeder):
        if len(events_image) == 1:
            print('Undersample for ' + event_stream_path)
            break
        for m in range(len(events_image)):
            classnames = classnames_list[class_idxs[m]]
            events_image_i = events_image[m].numpy().transpose(0,2,3,4,1) # B,T,H,W,C
            B, T, H, W, C = events_image_i.shape
            for i in range(B):
                for j in range(T):
                    # TODO train
                    file_path = event_stream_path[0].replace('DvsGesture', 'DVSGesture_Sampled_train').replace('.aedat', '')
                    # TODO val
                    # file_path = event_stream_path[0].replace('DvsGesture', 'DVSGesture_Sampled_val').replace('.aedat','')

                    folder_path = '/'.join([file_path.split('/')[i] for i in range(7)])
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    class_path = '/'.join([folder_path, classnames])
                    # print(class_path)
                    if not os.path.exists(class_path):
                        os.makedirs(class_path)
                    file_path = class_path + '/' + file_path.split('/')[7]
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    subfile_path = file_path + '/' + str(j) +'.png'
                    print(subfile_path)
                    img = events_image_i[i,j,:,:,:]
                    cv2.imwrite(subfile_path, img)

