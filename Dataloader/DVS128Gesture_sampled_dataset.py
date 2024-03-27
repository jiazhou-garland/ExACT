from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy, json, cv2
import numpy as np
import matplotlib.pyplot as plt

class DVS128Gesture_sampled(Dataset):
    def __init__(self, txtPath, tfPath):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)
        tf = open(tfPath, "r")
        self.classnames_dict = json.load(tf)  # class name idx start from 0
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx:
        :return: events_image 3,T,H,W
                 image 3,H,W
                 label_idx 0 to cls 1
        """
        event_stream_path = self.files[idx].split('\t')
        events = []
        for i in range(len(event_stream_path)-1):
            event_frame = cv2.imread(event_stream_path[i])
            # print(event_stream_path[i])
            events.append(np.array(event_frame))
        events = np.array(events)

        if events.ndim < 4:
            events = events[np.newaxis,:,:,:]

        events_data = np.array(events).transpose(3,0,1,2) / 255.0
        # print(events_data)
        # events_data = torch.from_numpy(events_data)
        # print(event_stream_path)

        # label
        label_idx = self.classnames_dict[event_stream_path[0].split('/')[-3]]
        # print(label_idx)

        return events_data, label_idx

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        # random.shuffle(self.files)
        return len(self.files)

def pad_event(event, max_event_length):
    C, N, H, W = event.shape
    pad_num = max_event_length - N
    if pad_num > 0:
        pad_zeros = np.zeros((C, pad_num, H, W))
        event = np.concatenate((event, pad_zeros), axis=1)

    return event

def event_sampled_frames_collate_func(data):
    """
    Pad event data with various number of sampled frames among a batch.

    """
    events = [data[i][0] for i in range(len(data))]
    actual_event_length = [events[i].shape[1] for i in range(len(events))]
    max_event_length = max(actual_event_length)
    padded_events = np.array([pad_event(events[i], max_event_length) for i in range(len(events))])
    labels = np.array([data[i][1] for i in range(len(data))])
    actual_event_length = np.array(actual_event_length)

    return padded_events, actual_event_length, labels

def visualize_img(grayscale_img, classnames_list, labels):
    # print(image.shape)
    B,T,H,W,C = grayscale_img.shape
    plt_num = int((T + 1) / 5 + 1)
    for j in range(B):
        plt.figure()
        for i in range(T):
            img = grayscale_img[j, i, :, :, :]
            plt.subplot(plt_num, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            # plt.title('Frame No.'+str(i), loc='center')
        class_name = classnames_list[labels[j]]
        plt.title(class_name, loc='center')
        plt.savefig('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/ExACT_original/Dataloader/DVS128Gesture/test2.jpg')
        plt.show()

if __name__ == '__main__':
    # test code
    train_path = r"/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/ExACT_original/Dataloader/DVS128Gesture/DVS128Gesture_sampled_train.txt"
    tfPath = r"/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/ExACT_original/Dataloader/DVS128Gesture/DVS128Gesture.json"
    tf = open(tfPath, "r")
    classnames_dict = json.load(tf)  # class name idx start from 0
    classnames_list = [i for i in classnames_dict.keys()]
    datasets = DVS128Gesture_sampled(train_path, tfPath)
    feeder = DataLoader(datasets, batch_size=2, shuffle=True,
                        collate_fn=event_sampled_frames_collate_func)
    padded_events, actual_event_length, labels = next(iter(feeder))
    padded_events = padded_events.transpose(0,2,3,4,1) # B,T,H,W,C
    print(padded_events.shape)
    print(actual_event_length)
    visualize_img(padded_events, classnames_list, labels)

    # visualize_events_stream(events_stream)
    # visualize_events_image(events_stream, frame=frame)
    # print(events_image.shape)
    # analysis_dataset(train_path)

