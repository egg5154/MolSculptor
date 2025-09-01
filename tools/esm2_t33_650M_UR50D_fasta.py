from transformers import AutoTokenizer, EsmModel
import torch # type: ignore
import numpy as np
import argparse
import pickle as pkl 
from Bio import SeqIO
import datetime

import os
import sys

def arg_parse():
    parser = argparse.ArgumentParser(description='ESMfold')
    parser.add_argument('--fasta_path', type=str, required=True, help='Amino acid sequence')
    parser.add_argument('--output_pkl_path', type=str, required=True, help='Output file name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--model_path', type=str, help='Path to the pretrained model')
    return parser.parse_args()

args = arg_parse()
batch_size = args.batch_size
model_path = args.model_path
add_special_tokens = True

def main():
    fasta_sequences = SeqIO.parse(open(args.fasta_path),'fasta')
    sequence_key = []
    sequences = []
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequence_key.append(name)
        sequences.append(sequence)
    n_seqs = len(sequence_key)
    n_batches = n_seqs // batch_size + (n_seqs % batch_size != 0)
    sequence_key = sequence_key + [sequence_key[-1]] * (n_batches * batch_size - n_seqs)
    sequences = sequences + [sequences[-1]] * (n_batches * batch_size - n_seqs)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    embeddings = []
    for b in range(n_batches):
        inputs = tokenizer(sequences[b*batch_size:(b+1)*batch_size], return_tensors="pt", padding=True, add_special_tokens=add_special_tokens).to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state[:, 1:, :].cpu().detach().numpy()
        
        for i in range(last_hidden_states.shape[0]):
            embeddings.extend([
                emb[:len(seq), :].tolist()
                for emb, seq in zip(last_hidden_states, sequences[b*batch_size:(b+1)*batch_size])
            ])
    
    with open(args.output_pkl_path, 'wb') as f:
        pkl.dump(
            {
                # k: {'sequence': s, 'embedding': np.mean(emb, axis=0)}
                k: {'sequence': s, 'embedding': np.array(emb),}
                for k, s, emb in zip(sequence_key, sequences, embeddings)
            }, f
        )
    
    print("esm2_t33_650M_UR50D embeddings for input fasta file {} saved to {}".format(args.fasta_path, args.output_pkl_path))

if __name__ == "__main__":
    t0 = datetime.datetime.now()
    main()
    t1 = datetime.datetime.now()
    print('time used:', t1-t0)