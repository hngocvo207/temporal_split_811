import logging
import os
import os.path as osp
import time

import numpy as np
import torch

from torch_geometric.data import HeteroData

from fraudGT.datasets.mg_dataset import MGDataset
from fraudGT.datasets.eth_dataset import ETHDataset
from fraudGT.datasets.temporal_dataset import TemporalDataset
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.loader import set_dataset_attr
from fraudGT.graphgym.register import register_loader
from fraudGT.transform.posenc_stats import compute_posenc_stats

from torch_geometric.utils import index_to_mask
from fraudGT.loader.split_generator import prepare_splits, set_dataset_splits
from fraudGT.loader.encoding_generator import (
    preprocess_Node2Vec, check_Node2Vec, load_Node2Vec,
    preprocess_Metapath, check_Metapath, load_Metapath,
    preprocess_KGE, check_KGE, load_KGE,
)


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'num_nodes_dict'):
        total_num_nodes = sum(dataset.data.num_nodes_dict.values())
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: {total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(
        f"  num edge features: {dataset.num_edge_features} "
        f"(add_ports={cfg.dataset.add_ports})"
    )

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """Master loader for the MG-only clean pipeline."""
    if format in ('MG',):
        dataset_dir = osp.join(dataset_dir, 'MG')
        dataset = preformat_MG(dataset_dir)
    elif format in ('ETH',):
        dataset = preformat_ETH(dataset_dir)
    else:
        raise ValueError(f"Unknown data format: {format}. Supported: 'MG', 'ETH'.")

    log_loaded_dataset(dataset, format, name)

    # Positional encoding pre-computation (Node2Vec / Metapath / KGE).
    if cfg.posenc_Hetero_Node2Vec.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Node2Vec(pe_dir):
            preprocess_Node2Vec(pe_dir, dataset)
        model = load_Node2Vec(pe_dir)
        if isinstance(dataset, TemporalDataset):
            for split_idx, split in enumerate(['train', 'val', 'test']):
                homo_data = dataset[split].to_homogeneous()
                for idx, node_type in enumerate(dataset[split].node_types):
                    mask = homo_data.node_type == idx
                    dataset[split][node_type]['pestat_Hetero_Node2Vec'] = model[split_idx][mask]

    if cfg.posenc_Hetero_Metapath.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_Metapath(pe_dir):
            preprocess_Metapath(pe_dir, dataset)
        model = load_Metapath(pe_dir)
        emb = model['model'].weight.data.detach().cpu()
        if hasattr(dataset, 'dynamicTemporal'):
            data_list = []
            for split in range(3):
                data = dataset[split]
                for node_type in dataset.data.node_types:
                    data[node_type]['pestat_Hetero_Metapath'] = \
                        emb[model['start'][node_type]:model['end'][node_type]]
                data_list.append(data)
            dataset._data, dataset.slices = dataset.collate(data_list)

    if cfg.posenc_Hetero_TransE.enable:
        pe_dir = osp.join(dataset_dir, name.replace('-', '_'), 'posenc')
        if not osp.exists(pe_dir) or not check_KGE(pe_dir, 'TransE'):
            preprocess_KGE(pe_dir, dataset, 'TransE')
        model = load_KGE(pe_dir, 'TransE', dataset)
        for node_type in dataset.data.num_nodes_dict:
            dataset.data[node_type]['pestat_Hetero_TransE'] = model[node_type].detach().cpu()

    print(dataset[0])

    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    prepare_splits(dataset)

    return dataset


def preformat_MG(dataset_dir):
    dataset = MGDataset(
        root=dataset_dir,
        dynamic_dir=cfg.dataset.dynamic_dir,
        ratio=cfg.dataset.ratio,
        reverse_mp=cfg.dataset.reverse_mp,
        add_ports=cfg.dataset.add_ports,
    )
    return dataset


def preformat_ETH(dataset_dir):
    dataset = ETHDataset(
        root=dataset_dir,
        reverse_mp=cfg.dataset.reverse_mp,
        add_ports=cfg.dataset.add_ports,
    )
    return dataset


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object."""
    assert len(datasets) == 3, "Expecting train, val, test datasets"
    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = (
        [datasets[0].get(i) for i in range(n1)]
        + [datasets[1].get(i) for i in range(n2)]
        + [datasets[2].get(i) for i in range(n3)]
    )
    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    datasets[0].split_idxs = [
        list(range(n1)),
        list(range(n1, n1 + n2)),
        list(range(n1 + n2, n1 + n2 + n3)),
    ]
    return datasets[0]
