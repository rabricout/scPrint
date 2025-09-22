A repo to try to use scPRINT

## Usage

In scPrint/, launch : 
'python my_tests/embedding.py'
to create the embeddings of a given h5ad file. If there are some gene missmatches, a "missing_genes_ids.json" file will be created. 

In scPrint/, launch : 
'python my_tests/grn_infer.py'
to create grn from the embeddings created using "embedding.py". If needed, "missing_genes_ids.json" file will be used. 
