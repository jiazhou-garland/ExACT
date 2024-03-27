import os, random,json
from tqdm import tqdm
import numpy as np
def write_file(path, events):
    """
    save the list into a __.txt file
    """
    # events = sorted(events)
    with open(path, 'w') as f:
        for i in range(len(events)):
            f.write('{}\n'.format(events[i]))
        f.close()

def write_sampled_file(path, events):
    """
    save the list into a __.txt file
    """
    # events = sorted(events)
    T_max = 0
    with open(path, 'w') as f:
        for i in range(len(events)):
            subevents = events[i]
            skip = 1
            if len(subevents) >= 64 and len(subevents)<128:
                skip = 4
            if len(subevents) >= 32 and len(subevents)<64:
                skip = 2
            if len(subevents) >= 128:
                skip = 8
            if int(len(subevents)/skip) > T_max:
                T_max = int(len(subevents)/skip)
            if int(len(subevents)/skip) <= 1:
                print(subevents)
                pass
            for j in range(int(len(subevents)/skip)):
                f.write('{}\t'.format(subevents[skip*j]))
            f.write('\n')
        f.close()
    print(T_max)

def train_val_divide(train_csv, val_csv, event_path_dataset):
    """
    divide the train ana validation data,
    """
    print('开始划分训练集与验证集')
    events, train_e, val_e, class_dic = [], [], [], {}

    with open(train_csv, 'r') as file:
        # Read the entire contents of the file
        for line in file:
            event_str = event_path_dataset + line.strip()
            print(event_str)
            train_e.append(event_str)

    with open(val_csv, 'r') as file:
        # Read the entire contents of the file
        for line in file:
            event_str = event_path_dataset + line.strip()
            print(event_str)
            val_e.append(event_str)

    print("训练集与验证集划分结束")
    return train_e, val_e

def list_collection(event_path_dataset):
    """
    divide the train ana validation data,
    """
    print('开始划分训练集与验证集')
    events = []

    for cat_id, cat in enumerate(tqdm(os.listdir(event_path_dataset))):
        file_samples = os.listdir(os.path.join(event_path_dataset, cat))
        for fs in file_samples:
            subfile_samples = os.listdir(os.path.join(event_path_dataset, cat, fs))
            sub_events = []
            for i in range(len(subfile_samples)):
                event_str = str(os.path.join(event_path_dataset, cat, fs, str(i)+'.png'))
                # print(event_str)
                sub_events.append(event_str)
            events.append(sub_events)

    return events

if __name__ == "__main__":
    train_csv = "/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DvsGesture/trials_to_train.txt"
    val_csv = "/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DvsGesture/trials_to_test.txt"
    event_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DvsGesture/'

    # train_e, val_e = train_val_divide(train_csv, val_csv, event_path_dataset)
    # write_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/Gesture128DVS/DVS128Gesture_train.txt', train_e)
    # write_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/Gesture128DVS/DVS128Gesture_val.txt', val_e)

    # # 50000, 30
    # event_sampled_train_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_train/'
    # event_sampled_val_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_val/'
    # train_e = list_collection(event_sampled_train_path_dataset)
    # val_e = list_collection(event_sampled_val_path_dataset)
    # write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_train.txt', train_e)
    # write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_val.txt', val_e)
    #
    # # 25000, 30
    # event_sampled_train_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_train_v2/'
    # event_sampled_val_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_val_v2/'
    # train_e = list_collection(event_sampled_train_path_dataset)
    # val_e = list_collection(event_sampled_val_path_dataset)
    # write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_train.txt', train_e)
    # write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_val.txt', val_e)
    #
    # 15000, 30
    event_sampled_train_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_train_v3/'
    event_sampled_val_path_dataset = '/hpc2hdd/home/jiazhouzhou/jiazhouzhou/dataset/DVSGesture_Sampled_val_v3/'
    train_e = list_collection(event_sampled_train_path_dataset)
    val_e = list_collection(event_sampled_val_path_dataset)
    write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_train_v3.txt', train_e)
    write_sampled_file('/hpc2hdd/home/jiazhouzhou/jiazhouzhou/code/HumanECLIP/Dataloader/DVS128Gesture/DVS128Gesture_sampled_val_v3.txt', val_e)