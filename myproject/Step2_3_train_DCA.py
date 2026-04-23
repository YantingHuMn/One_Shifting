import pandas as pd
import scanpy as sc
import argparse
from dca.api import dca

parser = argparse.ArgumentParser(description='DCA denoising')
parser.add_argument('--input', type=str, required=True, help='Input feather file path')
parser.add_argument('--output', type=str, required=True, help='Output feather file path')
parser.add_argument('--ae_type', type=str, required=True, help='Type of dca')
parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')

args = parser.parse_args()

# DCA require cells x genes
df = pd.read_feather(args.input)
print(f"Input shape: {df.shape}")

pos_col = df.iloc[:, 0]
cell_names = pos_col.tolist()
gene_names = df.columns[1:].tolist()

mat = df.iloc[:, 1:].values  # cells x genes
print(f"Matrix shape (cells x genes): {mat.shape}")

adata = sc.AnnData(mat.astype('float32'))  # cells x genes
adata.obs_names = [f"cell_{i}" for i in range(mat.shape[0])]  # unique obs names
adata.var_names = gene_names  # genes
adata.obs['original_pos'] = cell_names  # save original pos

print(f"AnnData shape before DCA: {adata.shape}")

dca(adata,
    mode='denoise',
    ae_type=args.ae_type,
    epochs=args.epochs,
    batch_size=args.batch_size,
    early_stop=args.patience)

print(f"AnnData shape after DCA: {adata.shape}")

denoised = pd.DataFrame(adata.X, columns=adata.var_names.tolist())
denoised.insert(0, 'pos', adata.obs['original_pos'].values)

denoised.to_feather(args.output)
print(f"Done! Output saved to: {args.output}")
print(f"Output shape: {denoised.shape}")