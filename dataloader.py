# dataloader.py

import copy
import pickle
import torchvision
import pandas as pd
from torch.utils.data import Dataset
import sklearn.preprocessing as preprocessing

from models.model import *
from models.resnet import *
from models.discriminator import *
import torch.optim as optim


class PrivacyDataLoader:
    def __init__(self, train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset):
        self.train_data = train_data
        self.train_label = train_label
        self.train_sensitive_label = train_sensitive_label
        self.test_data = test_data, test_label
        self.test_label = test_label
        self.test_sensitive_label = test_sensitive_label
        self.trainset = trainset
        self.testset = testset


class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor):
        Dataset.__init__(self)
        assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.sensitive_tensor = sensitive_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class AdultDataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None):
        self.name = "adult"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)

    def number_encode_features(self, df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column].astype(str))
        return result, encoders

    def load(self):
        train_data = pd.read_csv(
            "data/adult/adult.data",
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
        train_data.tail()

        train_data, encoders = self.number_encode_features(train_data)
        train_data = np.asarray(train_data)
        train_label = train_data[:, 14]
        train_sensitive_label = train_data[:, 9]
        train_data = np.concatenate((train_data[:, 0:9], train_data[:, 10:14]), axis=1)

        test_data = pd.read_csv(
            "data/adult/adult.test",
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"],
                sep=r'\s*,\s*',
                engine='python',
                na_values="?")
        test_data.tail()
        test_data, encoders = self.number_encode_features(test_data)
        test_data = np.asarray(test_data)
        test_label = test_data[:, 14]
        test_sensitive_label = test_data[:, 9]
        test_data = np.concatenate((test_data[:, 0:9], test_data[:, 10:14]), axis=1)

        self.n_target_class = 2
        self.n_sensitive_class = 2
        self.embed_length = 2  # 64
        self.train_batch_size = 128
        self.test_batch_size = 1000
        self.max_epoch_encoder = 300
        self.max_epoch_discriminator = 300

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]
        self.train_data = torch.from_numpy(train_data).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)

        self.test_data = torch.from_numpy(test_data).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()

        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)

        self.net = AdultDatasetNet(num_classes=2, embed_length=self.embed_length)
        self.target_net = AdultDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.discriminator_net = AdultDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.adversary_net = AdultDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=5e-2,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))


class ExtendedYaleBDataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None):
        self.name = "yale"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)

    def load(self):
        data1 = pickle.load(open("data/yaleb/set_0.pdata", "rb"), encoding='latin1')
        data2 = pickle.load(open("data/yaleb/set_1.pdata", "rb"), encoding='latin1')
        data3 = pickle.load(open("data/yaleb/set_2.pdata", "rb"), encoding='latin1')
        data4 = pickle.load(open("data/yaleb/set_3.pdata", "rb"), encoding='latin1')
        data5 = pickle.load(open("data/yaleb/set_4.pdata", "rb"), encoding='latin1')
        test = pickle.load(open("data/yaleb/test.pdata", "rb"), encoding='latin1')
        train_data = np.concatenate((data1["x"], data2["x"], data3["x"], data4["x"], data5["x"]), axis=0)
        train_label = np.concatenate((data1["t"], data2["t"], data3["t"], data4["t"], data5["t"]), axis=0)
        train_sensitive_label = np.concatenate(
            (data1["light"], data2["light"], data3["light"], data4["light"], data5["light"]), axis=0)
        test_data = test["x"]
        test_label = test["t"]
        test_sensitive_label = test["light"]

        index = test_sensitive_label != 5
        test_label = test_label[index]
        test_sensitive_label = test_sensitive_label[index]
        test_data = test_data[index]

        self.n_target_class = 38
        self.n_sensitive_class = 5
        self.embed_length = 100  # 100  # 64 # 100  # 64
        self.train_batch_size = 16
        self.test_batch_size = 100
        self.max_epoch_encoder = 300
        self.max_epoch_discriminator = 300

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]
        self.train_data = torch.from_numpy(train_data).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)

        self.test_data = torch.from_numpy(test_data).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()

        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)

        self.net = ExtendedYaleBNet(num_classes=38, embed_length=self.embed_length)
        self.target_net = ExtendedYaleBAdversary(num_classes=38, embed_length=self.embed_length)
        self.discriminator_net = ExtendedYaleBAdversary(num_classes=5, embed_length=self.embed_length)
        self.adversary_net = ExtendedYaleBAdversary(num_classes=5, embed_length=self.embed_length)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=5e-4,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(), lr=0.0001, weight_decay=5e-4, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.0001, weight_decay=5e-4, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                                  lr=0.0001, weight_decay=5e-4, betas=(0.9, 0.999))


class GermanDataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None):
        self.name = "german"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)

    def load(self):
        data = np.loadtxt('data/german/german.data-numeric')
        np.random.shuffle(data)
        data[:, 24] = data[:, 24] - 1
        index = (data[:, 6] == 1) | (data[:, 6] == 3) | (data[:, 6] == 4)
        data[:, 6] = (index).astype(int)

        np.random.shuffle(data)
        train_data = np.concatenate((data[0:700, 0:8], data[0:700, 9:24]), axis=1)
        train_label = data[0:700, 24]
        train_sensitive_label = data[0:700, 6]

        test_data = np.concatenate((data[700:1000, 0:8], data[700:1000, 9:24]), axis=1)
        test_label = data[700:1000, 24]
        test_sensitive_label = data[700:1000, 6]

        self.n_target_class = 2
        self.n_sensitive_class = 2
        self.embed_length = 2
        self.train_batch_size = 64
        self.test_batch_size = 100
        self.max_epoch_encoder = 300
        self.max_epoch_discriminator = 300

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]
        self.train_data = torch.from_numpy(train_data).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)

        self.test_data = torch.from_numpy(test_data).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()

        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)
        self.net = GermanDatasetNet(num_classes=2, embed_length=self.embed_length)
        self.target_net = GermanDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.discriminator_net = GermanDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.adversary_net = GermanDatasetAdversary(num_classes=2, embed_length=self.embed_length)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=5e-4,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))


class CIFAR10DataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None,
                 embed_length=512):
        self.name = "cifar10"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)
        self.embed_length = embed_length

    def load(self):

        # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.n_target_class = 2
        self.n_sensitive_class = 10
        self.train_batch_size = 128
        self.test_batch_size = 1000
        self.max_epoch_encoder = 200
        self.max_epoch_discriminator = 200

        trainset = torchvision.datasets.CIFAR10('../data', download=True, train=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10('../data', download=True, train=False, transform=transform_test)

        self.train_data = torch.from_numpy(np.transpose(trainset.train_data, axes=(0, 3, 2, 1))).float()
        self.train_sensitive_label = torch.from_numpy(np.asarray(trainset.train_labels)).long()
        self.test_data = torch.from_numpy(np.transpose(testset.test_data, axes=(0, 3, 2, 1))).float()
        self.test_sensitive_label = torch.from_numpy(np.asarray(testset.test_labels)).long()
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

        self.nolife_set = np.array([0, 1, 8, 9])
        self.life_set = np.array([2, 3, 4, 5, 6, 7])

        target = np.zeros(self.train_size)
        nolife_set = np.in1d(self.train_sensitive_label.numpy(), self.nolife_set)
        life_set = np.in1d(self.train_sensitive_label.numpy(), self.life_set)
        target[nolife_set] = 0
        target[life_set] = 1

        target_test = np.zeros(self.test_size)
        nolife_set = np.in1d(self.test_sensitive_label.numpy(), self.nolife_set)
        life_set = np.in1d(self.test_sensitive_label.numpy(), self.life_set)
        target_test[nolife_set] = 0
        target_test[life_set] = 1

        self.train_label = torch.from_numpy(copy.deepcopy(target)).long()
        self.test_label = torch.from_numpy(copy.deepcopy(target_test)).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)
        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)
        if self.embed_length == 2:
            type_adv = 1
        else:
            type_adv = 3

        self.net = ResNet18(num_classes=2, embed_length=self.embed_length)
        self.target_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                      num_classes=self.n_target_class)
        self.discriminator_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                      num_classes=self.n_sensitive_class)
        self.adversary_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                  num_classes=self.n_sensitive_class)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-2,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(),
                                           lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                        lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))


class CIFAR100DataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None, embed_length=512):
        self.name = "cifar100"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)
        self.embed_length = embed_length
    def load(self):
        # CIFAR100_LABELS_LIST = [
        #     'apple'(0), 'aquarium_fish'(1), 'baby'(2), 'bear'(3), 'beaver'(4), 'bed'(5), 'bee'(6), 'beetle'(7),
        #     'bicycle'(8), 'bottle'(9), 'bowl'(10), 'boy'(11), 'bridge'(12), 'bus'(13), 'butterfly'(14), 'camel'(15),
        #     'can'(16), 'castle'(17), 'caterpillar'(18), 'cattle'(19), 'chair'(20), 'chimpanzee'(21), 'clock'(22),
        #     'cloud'(23), 'cockroach'(24), 'couch'(25), 'crab'(26), 'crocodile'(27), 'cup'(28), 'dinosaur'(29),
        #     'dolphin'(30), 'elephant'(31), 'flatfish'(32), 'forest'(33), 'fox'(34), 'girl'(35), 'hamster'(36),
        #     'house'(37), 'kangaroo'(38), 'keyboard'(39), 'lamp'(40), 'lawn_mower'(41), 'leopard'(42), 'lion'(43),
        #     'lizard'(44), 'lobster'(45), 'man'(46), 'maple_tree'(47), 'motorcycle'(48), 'mountain'(49), 'mouse'(50),
        #     'mushroom'(51), 'oak_tree'(52), 'orange'(53), 'orchid'(54), 'otter'(55), 'palm_tree'(56), 'pear'(57),
        #     'pickup_truck'(58), 'pine_tree'(59), 'plain'(60), 'plate'(61), 'poppy'(62), 'porcupine'(63),
        #     'possum'(64), 'rabbit'(65), 'raccoon'(66), 'ray'(67), 'road'(68), 'rocket'(69), 'rose'(70),
        #     'sea'(71), 'seal'(72), 'shark'(73), 'shrew'(74), 'skunk'(75), 'skyscraper'(76), 'snail'(77), 'snake'(78),
        #     'spider'(79), 'squirrel'(80), 'streetcar'(81), 'sunflower'(82), 'sweet_pepper'(83), 'table'(84),
        #     'tank'(85), 'telephone'(86), 'television'(87), 'tiger'(88), 'tractor'(89), 'train'(90), 'trout'(91),
        #     'tulip'(92), 'turtle'(93), 'wardrobe'(94), 'whale'(95), 'willow_tree'(96), 'wolf'(97), 'woman'(98),
        #     'worm'(99)
        # ]

        self.aquatic_mammal = np.array([4, 30, 55, 72, 95])  # beaver, dolphin, otter, seal, whale
        self.fish = np.array([1, 32, 67, 73, 91])  # aquarium fish, flatfish, ray, shark, trout
        self.flowers = np.array([54, 62, 70, 82, 92])  # orchids, poppies, roses, sunflowers, tulips
        self.food_container = np.array([9, 10, 16, 28, 61])  # bottles, bowls, cans, cups, plates
        self.fruits_and_veg = np.array([0, 51, 53, 57, 83])  # apples, mushrooms, oranges, pears, sweet peppers
        self.household_electric = np.array([22, 39, 40, 86, 87])  # clock, computer keyboard, lamp, telephone, television
        self.household_furniture = np.array([5, 20, 25, 84, 94])  # bed, chair, couch, table, wardrobe
        self.insects = np.array([6, 7, 14, 18, 24])  # bee, beetle, butterfly, caterpillar, cockroach
        self.carnivores = np.array([3, 42, 43, 88, 97])  # bear, leopard, lion, tiger, wolf
        self.man_made_outdoor_things = np.array([12, 17, 37, 68, 76])  # bridge, castle, house, road, skyscraper
        self.natural_outdoor_things = np.array([23, 33, 49, 60, 71])  # cloud, forest, mountain, plain, sea
        self.omnivores_herbivors = np.array([15, 19, 21, 31, 38])  # camel, cattle, chimpanzee, elephant, kangaroo
        self.medium_size_mammals = np.array([34, 63, 64, 66, 75])  # fox, porcupine, possum, raccoon, skunk
        self.non_insect_invertebrates = np.array([26, 45, 77, 79, 99])  # crab, lobster, snail, spider, worm
        self.people = np.array([2, 11, 35, 46, 98])  # baby, boy, girl, man, woman
        self.reptiles = np.array([27, 29, 44, 78, 93])  # crocodile, dinosaur, lizard, snake, turtle
        self.small_mammals = np.array([36, 50, 65, 74, 80])  # hamster, mouse, rabbit, shrew, squirrel
        self.trees = np.array([47, 52, 56, 59, 96])  # maple, oak, palm, pine, willow
        self.vehicles1 = np.array([8, 13, 48, 58, 90])  # bicycle, bus, motorcycle, pickup truck, train
        self.vehicles2 = np.array([41, 69, 81, 85, 89])  # lawn-mower, rocket, streetcar, tank, tractor

        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.n_target_class = 20
        self.n_sensitive_class = 100
        if self.embed_length == 2:
            type_adv = 1
        else:
            type_adv = 3
        self.train_batch_size = 128
        self.test_batch_size = 100
        self.max_epoch_encoder = 100
        self.max_epoch_discriminator = 100

        trainset = torchvision.datasets.CIFAR100('data/', download=True, train=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100('data/', download=True, train=False, transform=transform_test)

        self.train_data = torch.from_numpy(np.transpose(trainset.train_data, axes=(0, 3, 2, 1))).float()
        self.train_sensitive_label = torch.from_numpy(np.asarray(trainset.train_labels)).long()
        self.test_data = torch.from_numpy(np.transpose(testset.test_data, axes=(0, 3, 2, 1))).float()
        self.test_sensitive_label = torch.from_numpy(np.asarray(testset.test_labels)).long()
        self.train_size = self.train_data.shape[0]
        self.test_size = self.test_data.shape[0]

        aquatic_mammal = np.in1d(self.train_sensitive_label.numpy(), self.aquatic_mammal)
        fish = np.in1d(self.train_sensitive_label.numpy(), self.fish)
        flowers = np.in1d(self.train_sensitive_label.numpy(), self.flowers)
        food_container = np.in1d(self.train_sensitive_label.numpy(), self.food_container)
        fruits_and_veg = np.in1d(self.train_sensitive_label.numpy(), self.fruits_and_veg)
        household_electric = np.in1d(self.train_sensitive_label.numpy(), self.household_electric)
        household_furniture = np.in1d(self.train_sensitive_label.numpy(), self.household_furniture)
        insects = np.in1d(self.train_sensitive_label.numpy(), self.insects)
        carnivores = np.in1d(self.train_sensitive_label.numpy(), self.carnivores)
        man_made_outdoor_things = np.in1d(self.train_sensitive_label.numpy(), self.man_made_outdoor_things)
        natural_outdoor_things = np.in1d(self.train_sensitive_label.numpy(), self.natural_outdoor_things)
        omnivores_herbivors = np.in1d(self.train_sensitive_label.numpy(), self.omnivores_herbivors)
        medium_size_mammals = np.in1d(self.train_sensitive_label.numpy(), self.medium_size_mammals)
        non_insect_invertebrates = np.in1d(self.train_sensitive_label.numpy(), self.non_insect_invertebrates)
        people = np.in1d(self.train_sensitive_label.numpy(), self.people)
        reptiles = np.in1d(self.train_sensitive_label.numpy(), self.reptiles)
        small_mammals = np.in1d(self.train_sensitive_label.numpy(), self.small_mammals)
        trees = np.in1d(self.train_sensitive_label.numpy(), self.trees)
        vehicles1 = np.in1d(self.train_sensitive_label.numpy(), self.vehicles1)
        vehicles2 = np.in1d(self.train_sensitive_label.numpy(), self.vehicles2)

        target = np.zeros(self.train_size)
        target[aquatic_mammal] = 0
        target[fish] = 1
        target[flowers] = 2
        target[food_container] = 3
        target[fruits_and_veg] = 4
        target[household_electric] = 5
        target[household_furniture] = 6
        target[insects] = 7
        target[carnivores] = 8
        target[man_made_outdoor_things] = 9
        target[natural_outdoor_things] = 10
        target[omnivores_herbivors] = 11
        target[medium_size_mammals] = 12
        target[non_insect_invertebrates] = 13
        target[people] = 14
        target[reptiles] = 15
        target[small_mammals] = 16
        target[trees] = 17
        target[vehicles1] = 18
        target[vehicles2] = 19

        target_test = np.zeros(self.test_size)

        aquatic_mammal = np.in1d(self.test_sensitive_label.numpy(), self.aquatic_mammal)
        fish = np.in1d(self.test_sensitive_label.numpy(), self.fish)
        flowers = np.in1d(self.test_sensitive_label.numpy(), self.flowers)
        food_container = np.in1d(self.test_sensitive_label.numpy(), self.food_container)
        fruits_and_veg = np.in1d(self.test_sensitive_label.numpy(), self.fruits_and_veg)
        household_electric = np.in1d(self.test_sensitive_label.numpy(), self.household_electric)
        household_furniture = np.in1d(self.test_sensitive_label.numpy(), self.household_furniture)
        insects = np.in1d(self.test_sensitive_label.numpy(), self.insects)
        carnivores = np.in1d(self.test_sensitive_label.numpy(), self.carnivores)
        man_made_outdoor_things = np.in1d(self.test_sensitive_label.numpy(), self.man_made_outdoor_things)
        natural_outdoor_things = np.in1d(self.test_sensitive_label.numpy(), self.natural_outdoor_things)
        omnivores_herbivors = np.in1d(self.test_sensitive_label.numpy(), self.omnivores_herbivors)
        medium_size_mammals = np.in1d(self.test_sensitive_label.numpy(), self.medium_size_mammals)
        non_insect_invertebrates = np.in1d(self.test_sensitive_label.numpy(), self.non_insect_invertebrates)
        people = np.in1d(self.test_sensitive_label.numpy(), self.people)
        reptiles = np.in1d(self.test_sensitive_label.numpy(), self.reptiles)
        small_mammals = np.in1d(self.test_sensitive_label.numpy(), self.small_mammals)
        trees = np.in1d(self.test_sensitive_label.numpy(), self.trees)
        vehicles1 = np.in1d(self.test_sensitive_label.numpy(), self.vehicles1)
        vehicles2 = np.in1d(self.test_sensitive_label.numpy(), self.vehicles2)

        target_test[aquatic_mammal] = 0
        target_test[fish] = 1
        target_test[flowers] = 2
        target_test[food_container] = 3
        target_test[fruits_and_veg] = 4
        target_test[household_electric] = 5
        target_test[household_furniture] = 6
        target_test[insects] = 7
        target_test[carnivores] = 8
        target_test[man_made_outdoor_things] = 9
        target_test[natural_outdoor_things] = 10
        target_test[omnivores_herbivors] = 11
        target_test[medium_size_mammals] = 12
        target_test[non_insect_invertebrates] = 13
        target_test[people] = 14
        target_test[reptiles] = 15
        target_test[small_mammals] = 16
        target_test[trees] = 17
        target_test[vehicles1] = 18
        target_test[vehicles2] = 19

        self.train_label = torch.from_numpy(copy.deepcopy(target)).long()
        self.test_label = torch.from_numpy(copy.deepcopy(target_test)).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)
        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)

        self.net = ResNet18(num_classes=20, embed_length=self.embed_length)
        # self.net = Wide_ResNet(depth=28, widen_factor=10, dropout_rate=0.3, num_classes=20)
        self.target_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                      num_classes=self.n_target_class)

        self.discriminator_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                      num_classes=self.n_sensitive_class)
        self.adversary_net = create_discriminator(embed_length=self.embed_length, type=type_adv,
                                                  num_classes=self.n_sensitive_class)

        # self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-3, nesterov=False)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, weight_decay=1e-2,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(),
                                                  lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                                  lr=0.01, weight_decay=1e-3, betas=(0.9, 0.999))


class GaussianDataLoader(PrivacyDataLoader):
    def __init__(self, train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None):
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)
        self.name = "gaussian"

    def load(self):

        raw_input = np.loadtxt('data/gaussian2d/gaussian2d_input.txt')
        raw_label_color = np.loadtxt('data/gaussian2d/gaussian2d_label_color.txt')
        raw_label_shape = np.loadtxt('data/gaussian2d/gaussian2d_label_shape.txt')
        random_index = np.random.permutation(np.arange(0, 600))
        raw_input = raw_input[random_index, :]
        raw_label_color = raw_label_color[random_index]
        raw_label_shape = raw_label_shape[random_index]

        all_label = np.zeros(600)
        l1 = (raw_label_shape == 0) & (raw_label_color == 0)
        l2 = (raw_label_shape == 0) & (raw_label_color == 1)
        l3 = (raw_label_shape == 1) & (raw_label_color == 0)
        l4 = (raw_label_shape == 1) & (raw_label_color == 1)

        self.all_data = raw_input
        self.label = np.zeros(self.all_data.shape[0])
        self.label[l1] = 0
        self.label[l2] = 1
        self.label[l3] = 2
        self.label[l4] = 3

        train_data = raw_input[0:500, :]
        train_sensitive_label = raw_label_color[0:500]  # self.label[0:500]  # raw_label_color[0:500]
        train_label = raw_label_shape[0:500]
        test_data = raw_input[500:600, :]
        test_sensitive_label = raw_label_color[500:600]  # self.label[500:600]  # raw_label_color[500:600]
        test_label = raw_label_shape[500:600]

        self.n_target_class = 2
        self.n_sensitive_class = 2  # int(1+np.max(sensitive_label))
        self.embed_length = 1
        self.train_batch_size = 64
        self.test_batch_size = 64
        self.max_epoch_encoder = 200
        self.max_epoch_discriminator = 200

        self.train_size = train_data.shape[0]
        self.test_size = test_data.shape[0]
        self.train_data = torch.from_numpy(train_data).float()
        self.train_label = torch.from_numpy(train_label).long()
        self.train_sensitive_label = torch.from_numpy(train_sensitive_label).long()

        self.trainset = MyDataset(self.train_data, self.train_label, self.train_sensitive_label)

        self.test_data = torch.from_numpy(test_data).float()
        self.test_label = torch.from_numpy(test_label).long()
        self.test_sensitive_label = torch.from_numpy(test_sensitive_label).long()

        self.testset = MyDataset(self.test_data, self.test_label, self.test_sensitive_label)

        self.net = GaussianNet(num_classes=self.n_target_class, embed_length=self.embed_length)
        self.target_net = SwissRollAdversary(num_classes=self.n_target_class, embed_length=self.embed_length)
        self.discriminator_net = SwissRollAdversary(num_classes=self.n_sensitive_class, embed_length=self.embed_length)
        self.adversary_net = SwissRollAdversary(num_classes=self.n_sensitive_class, embed_length=self.embed_length)

        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=5e-4,
                                    betas=(0.9, 0.999))
        self.target_optimizer = optim.Adam(self.target_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
        self.adv_optimizer = optim.Adam(self.adversary_net.parameters(),
                                                  lr=0.001, weight_decay=1e-3, betas=(0.9, 0.999))
