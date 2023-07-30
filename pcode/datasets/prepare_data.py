# -*- coding: utf-8 -*-
import os
import json
import multiprocessing

# import spacy
# from spacy.symbols import ORTH
# import torchtext
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from genericpath import isfile
import numpy as np

# from pcode.datasets.loader.imagenet_folder import define_imagenet_folder
# from pcode.datasets.loader.svhn_folder import define_svhn_folder
# from pcode.datasets.loader.epsilon_or_rcv1_folder import define_epsilon_or_rcv1_folder


"""the entry for classification tasks."""


def _get_cifar(name, root, split, transform, target_transform, download):
    is_train = split == "train"

    # decide normalize parameter.
    if name == "cifar10":
        dataset_loader = datasets.CIFAR10
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )
    elif name == "cifar100":
        dataset_loader = datasets.CIFAR100
        normalize = transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )

    # decide data type.
    if is_train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((32, 32), 4),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return dataset_loader(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_mnist(root, split, transform, target_transform, download):
    is_train = split == "train"

    if is_train:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    return datasets.MNIST(
        root=root,
        train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )

class Femnist_dataset(Dataset):
    def __init__(self, root, split, rank, world_size, is_global):
        self._data, self._label = [], []
        self.world_size = world_size
        if not is_global:
            # local data
            self._append_data(root, split, rank)
        else:
            # global data (all agents)
            # raise ValueError("disabled: to avoid high memory usage!")
            jobs = []
            lock = multiprocessing.Lock()
            queue = multiprocessing.Queue()
            for i in range(world_size):
                # non-mp
                # self._append_data(root, split, i)

                # mp 
                p = multiprocessing.Process(target=self._load_data_mp, args=(root, split, i, lock, queue))
                jobs.append(p)
                p.start()
            
            for j in jobs:
                loaded = queue.get()
                self._data += loaded["data"]
                self._label += loaded["label"]
            
            for j in jobs:
                j.join()
            
        self._data = torch.tensor(self._data).reshape(-1, 1, 28, 28)
        self._data = torch.nn.functional.pad(self._data, (2,2,2,2)) # zero-padding to 32x32 images
        self._label = torch.tensor(self._label)
        
    def _load_data_mp(self, root, split, rank, lock, shared_queue):
        fn = os.path.join(root, split, "all_data_%d_niid_0_keep_0_%s_9.json"%(rank, split))
        with open(fn) as json_f:
            local_data = json.load(json_f)
        _data = []
        _label = []
        for key in local_data["user_data"]:
            _data += local_data["user_data"][key]["x"]
            _label += local_data["user_data"][key]["y"]
        # get lock
        # lock.acquire()
        shared_queue.put({"data": _data, "label": _label})
        # release lock
        # lock.release()
        print("done loading global_dataset: {}".format(rank))

    
    def _append_data(self, root, split, rank):
        print("load global_dataset: {}".format(rank))
        fn = os.path.join(root, split, "all_data_%d_niid_0_keep_0_%s_9.json"%(rank, split))
        with open(fn) as json_f:
            local_data = json.load(json_f)
        for key in local_data["user_data"]:
            self._data += local_data["user_data"][key]["x"]
            self._label += local_data["user_data"][key]["y"]
    
    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]
    
    def __len__(self):
        return len(self._data)



def _get_leaf_femnist(root, split, rank, world_size, is_global):
    # TODO: normalization of data to Normal(0, 1)
    # if is_train:
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #     )
    # else:
    #     transform = transforms.Compose(
    #         [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #     )
    return Femnist_dataset(root, split, rank, world_size, is_global)



def _get_stl10(root, split, transform, target_transform, download):
    return datasets.STL10(
        root=root,
        split=split,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_svhn(root, split, transform, target_transform, download):
    is_train = split == "train"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    return define_svhn_folder(
        root=root,
        is_train=is_train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def _get_imagenet(conf, name, datasets_path, split):
    is_train = split == "train"
    root = (
        os.path.join(
            datasets_path, "lmdb" if "downsampled" not in name else "lmdb_32x32"
        )
        if conf.use_lmdb_data
        else datasets_path
    )

    if is_train:
        root = os.path.join(
            root, "train{}".format("" if not conf.use_lmdb_data else ".lmdb")
        )
    else:
        root = os.path.join(
            root, "val{}".format("" if not conf.use_lmdb_data else ".lmdb")
        )
    return define_imagenet_folder(
        name=name, root=root, flag=conf.use_lmdb_data, cuda=conf.graph.on_cuda
    )


def _get_epsilon_or_rcv1(root, name, split):
    root = os.path.join(root, "{}_{}.lmdb".format(name, split))
    return define_epsilon_or_rcv1_folder(root)


"""the entry for language modeling task."""


def _get_text(batch_first):
    spacy_en = spacy.load("en")
    spacy_en.tokenizer.add_special_case("<eos>", [{ORTH: "<eos>"}])
    spacy_en.tokenizer.add_special_case("<bos>", [{ORTH: "<bos>"}])
    spacy_en.tokenizer.add_special_case("<unk>", [{ORTH: "<unk>"}])

    def spacy_tok(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    TEXT = torchtext.data.Field(lower=True, tokenize=spacy_tok, batch_first=batch_first)
    return TEXT


def _get_nlp_lm_dataset(name, datasets_path, batch_first):
    TEXT = _get_text(batch_first)

    # Load and split data.
    if "wikitext2" in name:
        train, valid, test = torchtext.datasets.WikiText2.splits(
            TEXT, root=datasets_path
        )
    elif "ptb" in name:
        train, valid, test = torchtext.datasets.PennTreebank.splits(
            TEXT, root=datasets_path
        )
    else:
        raise NotImplementedError
    return TEXT, train, valid, test


"""the entry for different supported dataset."""


def get_dataset(
    conf,
    name,
    datasets_path,
    split="train",
    transform=None,
    target_transform=None,
    download=True,
    is_global=False
):
    # create data folder if it does not exist.
    root = os.path.join(datasets_path, name)

    if name == "cifar10" or name == "cifar100":
        return _get_cifar(name, root, split, transform, target_transform, download)
    elif name == "svhn":
        return _get_svhn(root, split, transform, target_transform, download)
    elif name == "mnist":
        return _get_mnist(root, split, transform, target_transform, download)
    elif name == "stl10":
        return _get_stl10(root, split, transform, target_transform, download)
    elif "imagenet" in name:
        return _get_imagenet(conf, name, datasets_path, split)
    elif name == "epsilon":
        return _get_epsilon_or_rcv1(root, name, split)
    elif name == "rcv1":
        return _get_epsilon_or_rcv1(root, name, split)
    elif name == "wikitext2" or name == "ptb":
        return _get_nlp_lm_dataset(name, datasets_path, batch_first=False)
    elif name == "femnist":
        return _get_leaf_femnist(root, split, conf.rank, conf.n_mpi_process, is_global)
    else:
        raise NotImplementedError


def get_tomshardware_dataset(rank, world_size, seed, train_size=1000):
    datasource = RegressionData(rank, world_size, bias=False, seed=seed, train_size=train_size)
    train_loader = DataLoader(Simple_dataset(datasource.train_x, datasource.train_y),  batch_size=train_size, shuffle=True)
    test_loader = DataLoader(Simple_dataset(datasource.test_x, datasource.test_y),  batch_size=datasource.test_x.shape[0], shuffle=False)
    return train_loader, test_loader

    
class Simple_dataset(Dataset):
    def __init__(self, x, y):
        self._data = torch.tensor(x, dtype=torch.double)
        self._label = torch.tensor(y, dtype=torch.double)
    
    def __getitem__(self, idx):
        return self._data[idx], self._label[idx]
    
    def __len__(self):
        return len(self._data)


def append_bias(x):
    n = x.shape[0]
    return np.concatenate((x, np.ones((n, 1))), axis=1)


class RegressionData():
    def __init__(self, split_id=None, split_total=0, bias=False, seed=0, train_size=50):
        self.df, self.df_train, self.df_test, self.name = ("/home/user/TomsHardware/TomsHardware.data", 
                                            "/home/user/TomsHardware/TomsHardware_normalized_train.data", 
                                            "/home/user/TomsHardware/TomsHardware_normalized_test.data", 
                                            "tomshardware") # https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+
#         self.df, self.name = "/home/cyyau/project/regression/insurance.csv", "krr-insurance"
#         self.df, self.name = ("/home/cyyau/project/regression/airfoil_self_noise.csv", 
#                               "/home/cyyau/project/regression/airfoil_self_noise_normalized.csv",
#                               "krr-airfoil") # https://archive.ics.uci.edu/ml/datasets/airfoil+self-noise
        if not isfile(self.df_train):
            if split_id is None or split_id == 0:
                print("preprocessing dataset, storing to {} and {}".format(self.df_train, self.df_test))
                self.__preprocessor()
                exit()
            else:
                raise ValueError("unprocessed data, waiting to handle normalization and splitting")
        
        self.train = np.genfromtxt(self.df_train, delimiter=',', dtype=np.double)
        self.test = np.genfromtxt(self.df_test, delimiter=',', dtype=np.double)

        if split_total != 0 and not split_id is None:
            self.__split(split_id, split_total)


        # self.__shuffle(seed) # shuffle train data

        self.train_x, self.train_y = self.train[:, :-1], self.train[:, -1]
        self.test_x, self.test_y = self.test[:, :-1], self.test[:, -1]

        self.__transfer_samples_from_train_to_test(1000) # we only allow 1000 train samples per agent.
        if train_size > 1000:
            raise ValueError("We train on small number of samples~")
        elif train_size < 1000:
            # dropping the extra training samples to limit the train size
            self.train_x = self.train_x[:train_size]
            self.train_y = self.train_y[:train_size]

        if bias:
            self.train_x = append_bias(self.train_x)
            self.test_x = append_bias(self.test_x)
        
        self.dim = self.train_x.shape[-1]
        self.idx = 0
        self.idx_bound = self.train_x.shape[0]
        print("{} training {} testing samples, {}-dimension".format(self.train_x.shape[0], self.test_x.shape[0], self.train_x.shape[1]))

        self.kernel = RF_NTK(self.dim, 2, [16,16], [16,16], seed=seed)
            
        self.train_x = self.kernel.transform(self.train_x)
        self.test_x = self.kernel.transform(self.test_x)
        # sys.exit()
        # test samples are separated among agents to save mem and computation
    

    def __transfer_samples_from_train_to_test(self, num_of_train_samples):
        n = num_of_train_samples
        self.test_x = np.r_[self.test_x, self.train_x[n:]]
        self.test_y = np.r_[self.test_y, self.train_y[n:]]

        self.train_x = self.train_x[:n]
        self.train_y = self.train_y[:n]


    def __preprocessor(self, train_test_split=0.8):
        data = np.genfromtxt(self.df, delimiter=',')
        data_x, data_y = data[:, :-1], data[:, -1]
        data_x = self.__normalize(data_x)
        data_y = self.__preprocess_target(data_y) # either taking ln or scale to [-1, 1]
        # normalization is done on the whole network, with the same mean, std and max_y

        data = np.concatenate( (data_x, data_y.reshape(-1,1)), axis=-1)

        # split into training set and testing set
        total = len(data)
        split_idx = int(train_test_split*total)
        test_data = data[split_idx:]
        train_data = data[:split_idx]
        np.savetxt(self.df_train, train_data, delimiter=",")
        np.savetxt(self.df_test, test_data, delimiter=",")

    
    def __split(self, aid, split_total):
        # evenly split
        m = self.train.shape[0] // split_total
        self.train = self.train[aid*m: (aid+1)*m]
    
        m = self.test.shape[0] // split_total
        self.test = self.test[aid*m: (aid+1)*m]

        
    def __shuffle(self, seed):
        np.random.seed(seed)
        order = np.random.permutation(self.train.shape[0])
        self.train = self.train[order]
        

    def __normalize(self, train_x):
        mean = train_x.mean(axis=0)
        std = train_x.std(axis=0)
        train_x = (train_x - mean) / std
        return train_x
    

    def __preprocess_target(self, train_y, log=False, scale=True):
        if log:
            train_y = np.log(train_y)
        elif scale:
            max_abs = [np.abs(np.max(train_y)), np.abs(np.min(train_y))]
            max_abs = np.max(max_abs)
            train_y = train_y / max_abs
            self.scale_factor = max_abs
        return train_y
        
    
    def next_idx(self, size=32):
        btm = self.idx
        if self.idx + size >= self.idx_bound:
            top = self.idx_bound
            rem = self.idx + size - self.idx_bound
            self.idx = rem
            return list(range(btm, top)) + list(range(0, self.idx))
        else:
            self.idx += size
            return list(range(btm, self.idx))


class RF_NTK():
    def __init__(self, dimension, layers, feature0_dims, feature1_dims, seed=0):
        self.dim = dimension
        self.layers = layers
        self.m0s = [dimension] + feature0_dims
        self.m1s = [dimension] + feature1_dims
        self.seed = seed
        self.__sample_w( seed=self.seed)
    

    def __sample_w(self, seed=0):
        np.random.seed(seed + 100) # common seed
        self.W_0 = [np.random.normal(0, 1, (self.m0s[i+1], self.m0s[i])) for i in range(self.layers)]
        self.W_1 = [np.random.normal(0, 1, (self.m1s[i+1], self.m1s[i])) for i in range(self.layers)]


    def __acti_0(self, x, layer):
        act = np.heaviside(self.W_0[layer] @ x, 0)
        return act * np.sqrt(2 / self.m0s[layer+1])


    def __acti_1(self, x, layer):
        trans = self.W_1[layer] @ x
        act = trans * (trans > 0) # relu
        return act * np.sqrt(2 / self.m1s[layer+1])


    def __phi(self, x):
        phi_l = x
        psi_l = x
        for lay in range(self.layers):
            psi_l_next = self.__acti_1(psi_l, lay)
            phi_l_next = np.r_[np.kron(self.__acti_0(psi_l, lay), phi_l), psi_l_next]

            psi_l = psi_l_next
            phi_l = phi_l_next
        
        return phi_l # NTK
        # return phi_l # NNGP ?


    def __phi_func(self, X):
        feature_ntk = []
        for x in X:
            feature_ntk.append(self.__phi(x))
        return np.array(feature_ntk)


    def transform(self, x):
        x = self.__phi_func(x)
        x = append_bias(x)
        return x
