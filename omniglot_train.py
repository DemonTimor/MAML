import os
import random
import time
import numpy as np
from numpy.core.fromnumeric import resize
import torch
from MetaLearner import MetaLearner
import  argparse
import torchvision.transforms as transforms
from PIL import Image

def load_data_cache(dataset, args):
    """
    Collects several batches data for N-shot learning
    :param dataset: [cls_num, 20, 84, 84, 1]
    :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
    """
    #  take 5 way 1 shot as example: 5 * 1
    k_spt = args.k_spt
    n_way = args.n_way
    k_query = args.k_query
    batch_size = args.task_num
    resize = args.imgsz
    setsz = k_spt * n_way
    querysz = k_query * n_way
    data_cache = []

    # print('preload next 10 caches of batch_size of batch.')
    for sample in range(50):  # num of epochs

        x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
        for i in range(batch_size):  # one batch means one set

            x_spt, y_spt, x_qry, y_qry = [], [], [], []
            selected_cls = np.random.choice(dataset.shape[0], n_way, replace =  False) 

            for j, cur_class in enumerate(selected_cls):

                selected_img = np.random.choice(20, k_spt + k_query, replace = False)

                # 构造support集和query集
                x_spt.append(dataset[cur_class][selected_img[:k_spt]])
                x_qry.append(dataset[cur_class][selected_img[k_spt:]])
                y_spt.append([j for _ in range(k_spt)])
                y_qry.append([j for _ in range(k_query)])

            # shuffle inside a batch
            perm = np.random.permutation(n_way * k_spt)
            x_spt = np.array(x_spt).reshape(n_way * k_spt, 1, resize, resize)[perm]
            y_spt = np.array(y_spt).reshape(n_way * k_spt)[perm]
            perm = np.random.permutation(n_way * k_query)
            x_qry = np.array(x_qry).reshape(n_way * k_query, 1, resize, resize)[perm]
            y_qry = np.array(y_qry).reshape(n_way * k_query)[perm]
 
            # append [sptsz, 1, 84, 84] => [batch_size, setsz, 1, 84, 84]
            x_spts.append(x_spt)
            y_spts.append(y_spt)
            x_qrys.append(x_qry)
            y_qrys.append(y_qry)

#         print(x_spts[0].shape)
        # [b, setsz = n_way * k_spt, 1, 84, 84]
        x_spts = np.array(x_spts).astype(np.float32).reshape(batch_size, setsz, 1, resize, resize)
        y_spts = np.array(y_spts).astype(np.int64).reshape(batch_size, setsz)
        # [b, qrysz = n_way * k_query, 1, 84, 84]
        x_qrys = np.array(x_qrys).astype(np.float32).reshape(batch_size, querysz, 1, resize, resize)
        y_qrys = np.array(y_qrys).astype(np.int64).reshape(batch_size, querysz)
#         print(x_qrys.shape)
        data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

    return data_cache

def next(args, indexes, datasets, datasets_cache, mode='train'):
    """
    Gets next batch from the dataset with name.
    :param mode: The name of the splitting (one of "train", "val", "test")
    :return:
    """
    # update cache if indexes is larger than len(data_cache)
    if indexes[mode] >= len(datasets_cache[mode]):
        indexes[mode] = 0
        datasets_cache[mode] = load_data_cache(datasets[mode], args)

    next_batch = datasets_cache[mode][indexes[mode]]
    indexes[mode] += 1

    return next_batch

def find_classes(root_dir_train):
    img_items = []
    for (root, dirs, files) in os.walk(root_dir_train): 
        for file in files:
            if (file.endswith("png")):
                r = root.split(os.sep)
                img_items.append((file, r[-2] + os.sep + r[-1], root))
    print("== Found %d items " % len(img_items))
    return img_items

## 构建一个词典{class:idx}
def index_classes(items):
    class_idx = {}
    count = 0
    for item in items:
        if item[1] not in class_idx:
            class_idx[item[1]] = count
            count += 1
    print('== Found {} classes'.format(len(class_idx)))
    return class_idx

def generate_temp(img_items, class_idx):
    temp = dict()
    for imgname, classes, dirs in img_items:
        img = '{0}{1}{2}'.format(dirs, os.sep, imgname)
        label = class_idx[classes]
        transform = transforms.Compose([lambda img: Image.open(img).convert('L'),
                                lambda img: img.resize((28,28)),
                                lambda img: np.reshape(img, (28,28,1)),
                                lambda img: np.transpose(img, [2,0,1]),
                                lambda img: img/255.
                                ])
        img = transform(img)
        if label in temp.keys():
            temp[label].append(img)
        else:
            temp[label] = [img]
    print('begin to generate omniglot.npy file')
    return temp
    ## 每个字符包含20个样本

def main(args):

    random.seed(1337)
    np.random.seed(1337)

    device = torch.device('cuda')

    meta = MetaLearner(args).to(device)

    root = os.path.join('..{0}datasets{0}omniglot{0}python{0}').format(os.sep)
    if not (os.path.isfile(os.path.join(root, 'omniglot_test.npy')) and os.path.isfile(os.path.join(root, 'omniglot_test.npy'))):
        # 数据预处理
        '''
        an example of img_items:
        ( '0709_17.png',
        'Alphabet_of_the_Magi/character01',
        './../datasets/omniglot/python/images_background/Alphabet_of_the_Magi/character01')
        '''

        root_dir_train = os.path.join(root, 'images_background')
        root_dir_test = os.path.join(root, 'images_evaluation')

        img_items_train =  find_classes(root_dir_train) # [(file1, label1, root1),..]
        img_items_test = find_classes(root_dir_test)

        class_idx_train = index_classes(img_items_train)
        class_idx_test = index_classes(img_items_test)

        temp_train = generate_temp(img_items_train, class_idx_train)
        temp_test = generate_temp(img_items_test, class_idx_test)

        img_list = []
        for label, imgs in temp_train.items():
            img_list.append(np.array(imgs))
        img_list = np.array(img_list).astype(np.float) # [[20 imgs],..., 1623 classes in total]
        print('data shape:{}'.format(img_list.shape)) # (964, 20, 1, 28, 28)
        np.save(os.path.join('..{0}datasets{0}omniglot{0}python{0}'.format(os.sep), 'omniglot_train.npy'), img_list)
        print('end.')

        img_list = []
        for label, imgs in temp_test.items():
            img_list.append(np.array(imgs))
        img_list = np.array(img_list).astype(np.float) # [[20 imgs],..., 1623 classes in total]
        print('data shape:{}'.format(img_list.shape)) # (659, 20, 1, 28, 28)

        np.save(os.path.join('..{0}datasets{0}omniglot{0}python{0}'.format(os.sep), 'omniglot_test.npy'), img_list)
        print('end.')

    img_list_train = np.load(os.path.join(root, 'omniglot_train.npy')) # (964, 20, 1, 28, 28)
    img_list_test = np.load(os.path.join(root, 'omniglot_test.npy')) # (659, 20, 1, 28, 28)

    x_train = img_list_train
    x_test = img_list_test
    # num_classes = img_list.shape[0]

    indexes = {"train": 0, "test": 0}
    datasets = {"train": x_train, "test": x_test}
    print("DB: train", x_train.shape, "test", x_test.shape)

    datasets_cache = {"train": load_data_cache(x_train, args),  # current epoch data cached
                       "test": load_data_cache(x_test, args)}

    for step in range(args.epochs):
        start = time.time()
        x_spt, y_spt, x_qry, y_qry = next(args=args, indexes=indexes, datasets=datasets, datasets_cache=datasets_cache, mode='train')
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        accs, loss = meta(x_spt, y_spt, x_qry, y_qry)
        end = time.time()
        if step % 100 == 0:
            print("epoch:" ,step)
            print(accs)
    #         print(loss)
            
        if step % 1000 == 0:
            accs = []
            for _ in range(1000//args.task_num):
                # db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = next(args=args, indexes=indexes, datasets=datasets, datasets_cache=datasets_cache, mode='test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
                
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = meta.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc)
            print('在mean process之前：',np.array(accs).shape)
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('测试集准确率:',accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epochs', type=int, help='epoch number', default=60001)#60001, 60001, 60001
    argparser.add_argument('--n_way', type=int, help='n way', default=5)#5, 20, 20
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)#1, 1, 5
    argparser.add_argument('--k_query', type=int, help='k shot for query set', default=15)#15, 15, 15
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)#28, 28, 28
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)#32, 16, 16
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)#0.001, 0.001, 0.0008
    argparser.add_argument('--base_lr', type=float, help='task-level inner update learning rate', default=0.1)#0.1, 0.1, 0.075
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)#5, 5, 5
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)#5, 5, 5

    args = argparser.parse_args()

    main(args)