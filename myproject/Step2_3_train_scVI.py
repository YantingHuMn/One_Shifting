import pandas as pd
import scanpy as sc
import scvi
import argparse

parser = argparse.ArgumentParser(description='scVI denoising')
parser.add_argument('--input', type=str, required=True, help='Input feather file path')
parser.add_argument('--output', type=str, required=True, help='Output feather file path')
parser.add_argument('--loss', type=str, required=True, help='zinb, nb, poisson')
parser.add_argument('--max_epochs', type=int, default=60, help='Max training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

args = parser.parse_args()

# SCVI require gene by cell
df = pd.read_feather(args.input)
cell_ids = df["pos"].astype(str)
df = df.drop(columns=["pos"])

adata = sc.AnnData(df)
adata.obs_names = cell_ids.to_numpy()
adata.layers["counts"] = adata.X.copy()

scvi.model.SCVI.setup_anndata(adata, layer="counts")
model = scvi.model.SCVI(adata, gene_likelihood=args.loss)

model.train(
    max_epochs=args.max_epochs,
    batch_size=args.batch_size,
    early_stopping=True,
    early_stopping_patience=args.patience,
    early_stopping_monitor="elbo_validation",
    train_size=0.9,  # 90% train, 10% val
    datasplitter_kwargs={"drop_last": True},
)

denoised = model.get_normalized_expression()

denoised_df = pd.DataFrame(
    denoised,
    index=adata.obs_names,
    columns=adata.var_names
)

denoised_df.insert(0, "pos", denoised_df.index)
denoised_df.reset_index(drop=True).to_feather(args.output)

print(f"Done! Output saved to: {args.output}")