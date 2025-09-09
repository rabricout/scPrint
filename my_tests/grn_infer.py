from scprint.tasks import Denoiser, Embedder, GNInfer
from scprint import scPrint

import scanpy as sc
import torch
import json


with open('/mnt/sda/DATASETS/Single_nucleus_RNA-seq_adult_human_kidney/missing_genes_ids.json', 'r') as f:
    missing = json.load(f)

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
if len(missing) > 0:
    print(
        "Warning: some genes missmatch exist between model and ontology: solving...",
    )
    model._rm_genes(missing)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    
grn_inferer = GNInfer(
    layer=[0, 1],
    batch_size=16,
    how="random expr",
    preprocess="softmax",
    head_agg="none",
    filtration="none",
    forward_mode="none",
    #num_genes=100,
    #max_cells=10,
    doplot=False,
)

adata = sc.read_h5ad("/mnt/sda/DATASETS/Single_nucleus_RNA-seq_adult_human_kidney/embedded_data.h5ad")

#print(adata.obs['cell_type'].value_counts())
#input('Press a key to continue')

adata_type = adata[adata.obs['cell_type'] == 'epithelial cell of proximal tubule']

grn = grn_inferer(model, adata_type)
grn.varp['all'] = grn.varp['GRN'].copy()
# now we aggregate the heads by taking their average
grn.varp['GRN'] = grn.varp['GRN'].mean(-1)

