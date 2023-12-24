from utils import load_data
from .dataset import LinkPrediction

def build_dataset(cfgs):

    loaded_data = (adjlists, edge_metapath_indices_list, matrix_s, type_mask,
     train_val_test_pos, train_val_test_neg) = load_data(cfgs.data.dataset, cfgs.data.dataset_root,
                                                         cfgs.model.name, cfgs.model.sub_gcn)

    train_pos = train_val_test_pos['train_pos' + cfgs.data.target_path]
    val_pos = train_val_test_pos['val_pos' + cfgs.data.target_path]
    test_pos = train_val_test_pos['test_pos' + cfgs.data.target_path]
    train_neg = train_val_test_neg['train_neg' + cfgs.data.target_path]
    val_neg = train_val_test_neg['val_neg' + cfgs.data.target_path]
    test_neg = train_val_test_neg['test_neg' + cfgs.data.target_path]

    train_set = LinkPrediction(train_pos, train_neg, 'train')
    val_set = LinkPrediction(val_pos, val_neg, 'val')
    test_set = LinkPrediction(test_pos, test_neg, 'test')

    return train_set, val_set, test_set, loaded_data
