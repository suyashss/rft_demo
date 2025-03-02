import pandas as pd
import numpy as np
from argparse import ArgumentParser
from datasets import Dataset

def create_joint_df(loci,labels):
    loci_df = pd.read_csv(loci,sep="\t")
    labels_df = pd.read_csv(labels,sep="\t")
    error_msg = f"Length mismatch between features={len(loci_df)},labels={len(labels_df)}"
    error_msg += f"\n. Files are {loci},{labels}"
    assert len(loci_df) == len(labels_df), error_msg
    combined_df = pd.concat([loci_df,labels_df],
                            axis='columns')
    return combined_df

def main():

    parser = ArgumentParser("Create datasets")
    parser.add_argument("--loci_files",required=True,nargs='+')
    parser.add_argument("--label_files",required=True,nargs='+')
    parser.add_argument("--hf_dataset",required=True)
    parser.add_argument("--min_ngenes",type=int,default=5)
    parser.add_argument("--max_ngenes",type=int,default=25)
    parser.add_argument("--hf_dataset_config",default="default")
    parser.add_argument("--test_fraction",type=float,default=0.67)

    np.random.seed(42)
    args = parser.parse_args()

    print(args.loci_files)
    print(args.label_files)
    all_df_list = [create_joint_df(loci,labels) for loci,labels in zip(args.loci_files,args.label_files)]
    all_df = pd.concat(all_df_list,axis='index')
    print(f"Number of examples overall={len(all_df)}")
    
    idx = np.arange(len(all_df))
    np.random.shuffle(idx)

    num_train = int(args.test_fraction*len(idx))
    train_df = all_df.iloc[:num_train]
    test_df = all_df.iloc[num_train:]

    train_df['ngenes'] = train_df['symbol_gene_string'].apply(lambda x:len(x.split(",")))
    test_df['ngenes'] = test_df['symbol_gene_string'].apply(lambda x:len(x.split(",")))
    print(f"Number of examples in train={len(train_df)}")
    print(f"Number of examples in test={len(test_df)}")

    train_df_rft = train_df.loc[(train_df.ngenes >= args.min_ngenes) & (train_df.ngenes <= args.max_ngenes)]
    print(f"Number of examples in train after genes filter={len(train_df_rft)}")
    test_df_rft = test_df.loc[(test_df.ngenes >= args.min_ngenes) & (test_df.ngenes <= args.max_ngenes)]
    print(f"Number of examples in test after genes filter={len(test_df_rft)}")

    exclude_mask = train_df_rft.symbol.isin(test_df_rft.symbol) 
    train_df_rft = train_df_rft.loc[~exclude_mask] 
    print(f"Number of examples in train after removing eval genes/phenotypes={len(train_df_rft)}")

    train_df_rft = train_df_rft.drop_duplicates(subset=['description','symbol_gene_string'])
    test_df_rft = test_df_rft.drop_duplicates(subset=['description','symbol_gene_string'])
    print(f"Number of examples in train after removing duplicates={len(train_df_rft)}")
    print(f"Number of examples in test after removing duplicates={len(test_df_rft)}")

    train_dataset = Dataset.from_pandas(train_df_rft)
    eval_dataset = Dataset.from_pandas(test_df_rft)

    train_dataset.push_to_hub(args.hf_dataset,
                            config_name=args.hf_dataset_config,
                            split='train')
    eval_dataset.push_to_hub(args.hf_dataset, 
                            config_name=args.hf_dataset_config,
                            split='eval')

    print(f"Uploaded to {args.hf_dataset}")

if __name__ == "__main__":
    main()