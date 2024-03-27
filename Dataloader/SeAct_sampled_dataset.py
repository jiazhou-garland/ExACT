from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

class SeAct_sampled(Dataset):
    def __init__(self, txtPath, tfPath):
        self.txtPath = txtPath
        self.files = []
        self.labels = []
        self.length = self._readTXT(self.txtPath)

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
            events.append(np.array(event_frame))

        events_data = np.array(events).transpose(3,0,1,2) / 255.0
        # events_data = torch.from_numpy(events_data)

        # label
        label_str = event_stream_path[0].split('/')[-2]
        label_idx = int(label_str)
        # print(event_stream_path)
        # print(label_idx)
        # print(events_data.shape)
        return events_data, label_idx

    def _readTXT(self, txtPath):
        with open(txtPath, 'r') as f:
            for line in f.readlines():
                self.files.append(line)
        # random.shuffle(self.files)
        return len(self.files)

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
        plt.show()

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

if __name__ == '__main__':
    # test code
    train_path = r".\PAF\PAF_sampled_train.txt"
    tfPath = r".\PAF\PAF.json"
    tf = open("/ExACT/Dataloader/PAF/PAF.json", "r")
    classnames_dict = json.load(tf)  # class name idx start from 0
    classnames_list = [i for i in classnames_dict.keys()]
    datasets = DVS_SA_sampled(train_path)
    feeder = DataLoader(datasets, batch_size=2, shuffle=True,
                        collate_fn=event_sampled_frames_collate_func)
    padded_events, actual_event_length, labels = next(iter(feeder))
    padded_events = padded_events.transpose(0,2,3,4,1) # B,T,H,W,C
    visualize_img(padded_events, classnames_list, labels)

    # visualize_events_stream(events_stream)
    # visualize_events_image(events_stream, frame=frame)
    # print(events_image.shape)
    # analysis_dataset(train_path)

