import pandas as pd
from argparse import ArgumentParser
from datasets import Dataset

def create_joint_df(loci,labels):
    loci_df = pd.read_csv(loci,sep="\t")
    labels_df = pd.read_csv(labels,sep="\t")
    combined_df = pd.concat([loci_df,labels_df],
                            axis='columns')
    return combined_df

def main():

    parser = ArgumentParser("Create datasets")
    parser.add_argument("--train_loci",required=True)
    parser.add_argument("--train_labels",required=True)
    parser.add_argument("--eval_loci",required=True)
    parser.add_argument("--eval_labels",required=True)
    parser.add_argument("--hf_dataset",required=True)
    parser.add_argument("--min_ngenes",type=int,default=5)
    parser.add_argument("--max_ngenes",type=int,default=25)
    parser.add_argument("--hf_dataset_config",default="default")

    args = parser.parse_args()

    train_df = create_joint_df(args.train_loci,
                               args.train_labels)
    
    eval_df = create_joint_df(args.eval_loci,
                              args.eval_labels)

    train_df['ngenes'] = train_df['symbol_gene_string'].apply(lambda x:len(x.split(",")))
    eval_df['ngenes'] = eval_df['symbol_gene_string'].apply(lambda x:len(x.split(",")))
    print(f"Number of examples in train={len(train_df)}")
    print(f"Number of examples in eval={len(eval_df)}")

    train_df_rft = train_df.loc[(train_df.ngenes >= args.min_ngenes) & (train_df.ngenes <= args.max_ngenes)]
    print(f"Number of examples in train after genes filter={len(train_df_rft)}")
    eval_df_rft = eval_df.loc[(eval_df.ngenes >= args.min_ngenes) & (eval_df.ngenes <= args.max_ngenes)]
    print(f"Number of examples in eval after genes filter={len(eval_df_rft)}")

    exclude_mask = train_df_rft.symbol.isin(eval_df_rft.symbol) 
    train_df_rft = train_df_rft.loc[~exclude_mask] 
    print(f"Number of examples in train after removing eval genes/phenotypes={len(train_df_rft)}")

    train_df_rft = train_df_rft.drop_duplicates(subset=['description','symbol_gene_string'])
    eval_df_rft = eval_df_rft.drop_duplicates(subset=['description','symbol_gene_string'])
    print(f"Number of examples in train after removing duplicates={len(train_df_rft)}")
    print(f"Number of examples in eval after removing duplicates={len(eval_df_rft)}")

    train_dataset = Dataset.from_pandas(train_df_rft)
    eval_dataset = Dataset.from_pandas(eval_df_rft)

    train_dataset.push_to_hub(args.hf_dataset,
                               config_name=args.hf_dataset_config,
                               split='train')
    eval_dataset.push_to_hub(args.hf_dataset, 
                             config_name=args.hf_dataset_config,
                             split='eval')

    print(f"Uploaded to {args.hf_dataset}")

if __name__ == "__main__":
    main()