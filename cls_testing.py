import numpy as np
import pandas as pd
import torch.cuda
import torch
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

from utils import Extractor, Classifier, GetFeatures

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def predict(test_features, test_target, input_dim, ext_hid_dim, cls_hid_dim, output_dim,extractor_path, classifier_path):
    extractor = Extractor.SharedExtractor(input_dim, ext_hid_dim, ext_hid_dim // 2).to(device)
    classifier = Classifier.TaskSpecificClassifier(ext_hid_dim // 2, cls_hid_dim // 4, output_dim).to(device)
    extractor.load_state_dict(torch.load(extractor_path))
    classifier.load_state_dict(torch.load(classifier_path))
    extractor.eval()
    classifier.eval()
    features = extractor(test_features)
    output = classifier(features)
    y_pred = F.softmax(output)
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = y_pred.argmax(axis=1)
    y_test = test_target.argmax(axis=1)
    for i, v in enumerate(y_test):
        if v != 5:
            y_test[i] = 0
        else:
            y_test[i] = 1
    for i, v in enumerate(y_pred):
        if v != 5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    acc = accuracy_score(y_pred=y_pred, y_true=y_test)
    pre = precision_score(y_pred=y_pred, y_true=y_test)
    rec = recall_score(y_pred=y_pred, y_true=y_test)
    mcc = matthews_corrcoef(y_pred=y_pred, y_true=y_test)
    print('=' * 100)
    print('Accuracy: {}'.format(acc))
    print('Precision score: {}'.format(pre))
    print('Recall score: {}'.format(rec))
    print('MCC score: {}'.format(mcc))



if __name__ == '__main__':
    input_dim = 9554
    ex_hid_dim = 200
    cls_hid_dim = 60
    output_dim = 6
    ext_path = './pretrained_model/extractor.pth' # Pretrained extractor model path
    acpp_path = './pretrained_model/acp_cls.pth' # Pretrained ACP classifier model path
    acp_features_path = './features/acp_features'
    namp_features_path = './features/namp_features'
    acp_features = pd.read_csv(acp_features_path).iloc[:, 1:]
    namp_features = pd.read_csv(namp_features_path).iloc[:, 1:]
    acp_target = [[0, 0, 0, 0, 0, 1] for _ in range(len(acp_features))]
    namp_target = [[1, 0, 0, 0, 0, 0] for _ in range(len(namp_features))]
    _, acp_test_features, _, acp_test_target = train_test_split(acp_features,acp_target,
                                                                random_state=2022,shuffle=True,
                                                                test_size=0.2)
    acp_test_features = torch.tensor(np.array(acp_test_features), dtype=torch.float32).to(device)
    acp_test_target = torch.tensor(acp_test_target)
    print("ACP predicting......")
    predict(acp_test_features, acp_test_target, input_dim, ex_hid_dim, cls_hid_dim, output_dim,
         ext_path, acpp_path)
    print('finished......')