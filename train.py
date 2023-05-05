import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from utils import Extractor, Classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_cls_loss(pred, ground_true):
    return F.nll_loss(F.log_softmax(pred), torch.argmax(ground_true.long(), axis=-1))

def train_model(train_loader, test_loader, extractor, classifier, ext_optimizer, cls_optimizer, max_acc,
                test_dataset_length, save_dir, tag):
    extractor.to(device).train()
    classifier.to(device).train()
    for i, (x_train, y_train) in enumerate(train_loader):
        y_train = y_train.to(device)
        features = extractor(x_train.to(device))
        output = classifier(features)
        loss = get_cls_loss(output, y_train)
        ext_optimizer.zero_grad()
        cls_optimizer.zero_grad()
        torch.autograd.backward(loss)
        cls_optimizer.step()
        ext_optimizer.step()

    correct = 0
    extractor.eval()
    classifier.eval()
    for i, (x_test, y_test) in enumerate(test_loader):
        features = extractor(x_test.to(device))
        output = classifier(features)
        y_pred = F.softmax(output)
        y_pred = y_pred.detach().cpu().numpy()
        y_true = y_test.detach().cpu().numpy()
        correct += np.equal(y_pred.argmax(axis=1), y_true.argmax(axis=1)).sum()

    current_acc = correct * 1.0 / test_dataset_length
    if current_acc > max_acc:
        max_acc = current_acc
        torch.save(extractor.state_dict(), os.path.join(save_dir,'extractor.pth'))
        if tag == 'ABP':
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'abp_cls.pth'))
        elif tag == 'AFP':
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'afp_cls.pth'))
        elif tag == 'AHP':
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'ahp_cls.pth'))
        elif tag == 'AVP':
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'avp_cls.pth'))
        elif tag == 'ACP':
            torch.save(classifier.state_dict(), os.path.join(save_dir, 'acp_cls.pth'))
        else:
            raise ValueError
    return max_acc

def train(abp_train_loader, afp_train_loader, ahp_train_loader, avp_train_loader, acp_train_loader,
          abp_test_loader, afp_test_loader, ahp_test_loader, avp_test_loader, acp_test_loader,
          abp_test_length, afp_test_length, ahp_test_length, avp_test_length, acp_test_length,
          in_shapes, ex_hid_shapes, cls_hid_shapes, out_shapes,
          epochs, lr, save_dir):
    extractor = Extractor.SharedExtractor(in_shapes, ex_hid_shapes, ex_hid_shapes // 2)
    abp_classifier = Classifier.TaskSpecificClassifier(ex_hid_shapes // 2, cls_hid_shapes // 4, out_shapes)
    afp_classifier = Classifier.TaskSpecificClassifier(ex_hid_shapes // 2, cls_hid_shapes // 4, out_shapes)
    ahp_classifier = Classifier.TaskSpecificClassifier(ex_hid_shapes // 2, cls_hid_shapes // 4, out_shapes)
    avp_classifier = Classifier.TaskSpecificClassifier(ex_hid_shapes // 2, cls_hid_shapes // 4, out_shapes)
    acp_classifier = Classifier.TaskSpecificClassifier(ex_hid_shapes // 2, cls_hid_shapes // 4, out_shapes)

    extractor_optim = torch.optim.Adam(extractor.parameters(), lr=lr)
    abp_classifier_optim = torch.optim.Adam(abp_classifier.parameters(), lr=lr)
    afp_classifier_optim = torch.optim.Adam(afp_classifier.parameters(), lr=lr)
    ahp_classifier_optim = torch.optim.Adam(ahp_classifier.parameters(), lr=lr)
    avp_classifier_optim = torch.optim.Adam(avp_classifier.parameters(), lr=lr)
    acp_classifier_optim = torch.optim.Adam(acp_classifier.parameters(), lr=lr)

    max_acc_1 = 0
    max_acc_2 = 0
    max_acc_3 = 0
    max_acc_4 = 0
    max_acc_5 = 0

    for e in tqdm(range(epochs)):
        max_acc_1 = train_model(abp_train_loader, abp_test_loader, extractor, abp_classifier, extractor_optim,
                                abp_classifier_optim, max_acc_1, abp_test_length, save_dir, 'ABP')
        max_acc_2 = train_model(afp_train_loader, afp_test_loader, extractor, afp_classifier, extractor_optim,
                                afp_classifier_optim, max_acc_2, afp_test_length, save_dir, 'AFP')
        max_acc_3 = train_model(ahp_train_loader, ahp_test_loader, extractor, ahp_classifier, extractor_optim,
                                ahp_classifier_optim, max_acc_3, ahp_test_length, save_dir, 'AHP')
        max_acc_4 = train_model(avp_train_loader, avp_test_loader, extractor, avp_classifier, extractor_optim,
                                avp_classifier_optim, max_acc_4, avp_test_length, save_dir, 'AVP')
        max_acc_5 = train_model(acp_train_loader, acp_test_loader, extractor, acp_classifier, extractor_optim,
                                acp_classifier_optim, max_acc_5, acp_test_length, save_dir, 'ACP')

if __name__ == '__main__':
    batch_size=32
    epochs=150
    lr=0.0001
    input_dim=9554
    ex_hid_dim=200
    cls_hid_dim=60
    output_dim=6

    save_dir='./pretrained_model/'
    abp_features_path = './features/abp_features'
    afp_features_path = './features/afp_features'
    ahp_features_path = './features/ahp_features'
    avp_features_path = './features/avp_features'
    acp_features_path = './features/acp_features'
    namp_features_path = './features/namp_features'

    # loading and calculating features of peptides

    # abp_load_path = './data/ABP_only.fa'
    # afp_load_path = './data/AFP_only.fa'
    # ahp_load_path = './data/AHP_only.fa'
    # avp_load_path = './data/AVP_only.fa'
    # acp_load_path = './data/ACP_only.fa'
    # namp_load_path = './data/non_AMP_20190413.fa'
    # abp_features, abp_target = GetFeatures.get_features(abp_load_path,'ABP')
    # afp_features, afp_target = GetFeatures.get_features(afp_load_path,'AFP')
    # ahp_features, ahp_target = GetFeatures.get_features(ahp_load_path,'AHP')
    # avp_features, avp_target = GetFeatures.get_features(avp_load_path,'AVP')
    # acp_features, acp_target = GetFeatures.get_features(acp_load_path,'ACP')
    # namp_feature, namp_target = GetFeatures.get_features(namp_load_path,'NAMP')

    # prepared features
    print('data processing ......')
    abp_features = pd.read_csv(abp_features_path).iloc[:,1:]
    afp_features = pd.read_csv(afp_features_path).iloc[:,1:]
    ahp_features = pd.read_csv(ahp_features_path).iloc[:,1:]
    avp_features = pd.read_csv(avp_features_path).iloc[:,1:]
    acp_features = pd.read_csv(acp_features_path).iloc[:,1:]
    namp_features = pd.read_csv(namp_features_path).iloc[:,1:]

    abp_target = [[0, 1, 0, 0, 0, 0] for _ in range(len(abp_features))]
    afp_target = [[0, 0, 1, 0, 0, 0] for _ in range(len(afp_features))]
    ahp_target = [[0, 0, 0, 1, 0, 0] for _ in range(len(ahp_features))]
    avp_target = [[0, 0, 0, 0, 1, 0] for _ in range(len(avp_features))]
    acp_target = [[0, 0, 0, 0, 0, 1] for _ in range(len(acp_features))]
    namp_target = [[1, 0, 0, 0, 0, 0] for _ in range(len(namp_features))]

    abp_train_features, abp_test_features, abp_train_target, abp_test_target = train_test_split(abp_features,
                                                                                                abp_target,
                                                                                                random_state=2022,
                                                                                                shuffle=True,
                                                                                                test_size=0.2)
    afp_train_features, afp_test_features, afp_train_target, afp_test_target = train_test_split(afp_features,
                                                                                                afp_target,
                                                                                                random_state=2022,
                                                                                                shuffle=True,
                                                                                                test_size=0.2)
    ahp_train_features, ahp_test_features, ahp_train_target, ahp_test_target = train_test_split(ahp_features,
                                                                                                ahp_target,
                                                                                                random_state=2022,
                                                                                                shuffle=True,
                                                                                                test_size=0.2)
    avp_train_features, avp_test_features, avp_train_target, avp_test_target = train_test_split(avp_features,
                                                                                                avp_target,
                                                                                                random_state=2022,
                                                                                                shuffle=True,
                                                                                                test_size=0.2)
    acp_train_features, acp_test_features, acp_train_target, acp_test_target = train_test_split(acp_features,
                                                                                                acp_target,
                                                                                                random_state=2022,
                                                                                                shuffle=True,
                                                                                                test_size=0.2)
    abp_train_dataset = TensorDataset(
        torch.tensor(np.concatenate([abp_train_features, namp_features[:3500]]),dtype=torch.float32),
        torch.tensor(np.concatenate([abp_train_target, namp_target[:3500]])))
    afp_train_dataset = TensorDataset(
        torch.tensor(np.concatenate([afp_train_features, namp_features[:3500]]), dtype=torch.float32),
        torch.tensor(np.concatenate([afp_train_target, namp_target[:3500]])))
    ahp_train_dataset = TensorDataset(
        torch.tensor(np.concatenate([ahp_train_features, namp_features[:3500]]), dtype=torch.float32),
        torch.tensor(np.concatenate([ahp_train_target, namp_target[:3500]])))
    avp_train_dataset = TensorDataset(
        torch.tensor(np.concatenate([avp_train_features, namp_features[:3500]]), dtype=torch.float32),
        torch.tensor(np.concatenate([avp_train_target, namp_target[:3500]])))
    acp_train_dataset = TensorDataset(
        torch.tensor(np.concatenate([acp_train_features, namp_features[:3500]]), dtype=torch.float32),
        torch.tensor(np.concatenate([acp_train_target, namp_target[:3500]])))

    abp_test_dataset = TensorDataset(
        torch.tensor(np.concatenate([abp_test_features, namp_features[3500:]]), dtype=torch.float32),
        torch.tensor(np.concatenate([abp_test_target, namp_target[3500:]])))
    afp_test_dataset = TensorDataset(
        torch.tensor(np.concatenate([afp_test_features, namp_features[3500:]]), dtype=torch.float32),
        torch.tensor(np.concatenate([afp_test_target, namp_target[3500:]])))
    ahp_test_dataset = TensorDataset(
        torch.tensor(np.concatenate([ahp_test_features, namp_features[3500:]]), dtype=torch.float32),
        torch.tensor(np.concatenate([ahp_test_target, namp_target[3500:]])))
    avp_test_dataset = TensorDataset(
        torch.tensor(np.concatenate([avp_test_features, namp_features[3500:]]), dtype=torch.float32),
        torch.tensor(np.concatenate([avp_test_target, namp_target[3500:]])))
    acp_test_dataset = TensorDataset(
        torch.tensor(np.concatenate([acp_test_features, namp_features[3500:]]), dtype=torch.float32),
        torch.tensor(np.concatenate([acp_test_target, namp_target[3500:]])))

    abp_train_loader = DataLoader(abp_train_dataset, shuffle=True, batch_size=batch_size)
    afp_train_loader = DataLoader(afp_train_dataset, shuffle=True, batch_size=batch_size)
    ahp_train_loader = DataLoader(ahp_train_dataset, shuffle=True, batch_size=batch_size)
    avp_train_loader = DataLoader(avp_train_dataset, shuffle=True, batch_size=batch_size)
    acp_train_loader = DataLoader(acp_train_dataset, shuffle=True, batch_size=batch_size)

    abp_test_loader = DataLoader(abp_test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    afp_test_loader = DataLoader(afp_test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    ahp_test_loader = DataLoader(ahp_test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    avp_test_loader = DataLoader(avp_test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
    acp_test_loader = DataLoader(acp_test_dataset, shuffle=True, batch_size=batch_size, drop_last=True)

    print('training ......')
    train(abp_train_loader, afp_train_loader, ahp_train_loader, avp_train_loader, acp_train_loader, abp_test_loader,
          afp_test_loader, ahp_test_loader, avp_test_loader, acp_test_loader, len(abp_test_dataset), len(afp_test_dataset),
          len(ahp_test_dataset), len(avp_test_dataset), len(acp_test_dataset), input_dim, ex_hid_dim, cls_hid_dim, output_dim,
          epochs, lr, save_dir)

    print('training finished.')