import os
import urllib.request

import numpy as np
#import pytest
import scanpy as sc
import torch
from scdataloader import Preprocessor
from scdataloader.utils import load_genes
from scdataloader.utils import populate_my_ontology

import json

from scprint import scPrint
from scprint.base import NAME
from scprint.tasks import Denoiser, Embedder, GNInfer

import torch
torch.set_float32_matmul_precision('medium')

adata = sc.read_h5ad("/mnt/sda/DATASETS/Single_nucleus_RNA-seq_adult_human_kidney/1568e555-7c47-4b32-9f29-cda5717e9186.h5ad")
adata.obs.drop(columns="is_primary_data", inplace=True)
adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
preprocessor = Preprocessor(
    do_postp=False
)
adata = preprocessor(adata)

ckpt_path = "models/v2-medium.ckpt"
m = torch.load(ckpt_path)
transformer = "flash" if torch.cuda.is_available() else "normal"
model = scPrint.load_from_checkpoint(
    ckpt_path,
    precpt_gene_emb=None,
    classes=m['hyper_parameters']['label_counts'], 
    # triton gets installed so it must think it has cuda enabled
    transformer=transformer,
)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")


# this might happen if you have a model that was trained with a different set of genes than the one you are using in the ontology (e.g. newer ontologies), While having genes in the onlogy not in the model is fine. the opposite is not, so we need to remove the genes that are in the model but not in the ontology
missing = set(model.genes) - set(load_genes(model.organisms).index)
if len(missing) > 0:
    print(
        "Warning: some genes missmatch exist between model and ontology: solving...",
    )
    model._rm_genes(missing)
    with open('/mnt/sda/DATASETS/Single_nucleus_RNA-seq_adult_human_kidney/missing_genes_ids.json', 'w') as f:
        json.dump(list(missing), f)

input('OK')
        
# you can perform your inference on float16 if you have a GPU, otherwise use float64
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

embedder = Embedder( 
    # can work on random genes or most variables etc..
    how="random expr",
    # number of genes to use
    max_len=4000, 
    # the model is trained on 64 genes, so we will use this as a default
    batch_size=32,
    # for the dataloading
    num_workers=8, 
    # we will only use the cell type embedding here.
    #doclass=True,
    pred_embedding = ["cell_type_ontology_term_id"],
    #, "disease_ontology_term_id"],
    # we will now
    save_every=40_000,
    dtype=dtype,
)

adata, metrics = embedder(model, adata, cache=False)

adata.write_h5ad("/mnt/sda/DATASETS/Single_nucleus_RNA-seq_adult_human_kidney/embedded_data.h5ad")

