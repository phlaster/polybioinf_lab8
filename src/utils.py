import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import multiprocessing
from colorama import Fore, Style, init
init(autoreset=True)

def cout(text, color):
    color_dict = {
        'black': Fore.BLACK,
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
        'reset': Style.RESET_ALL
    }
    color_code = color_dict.get(color.lower(), Style.RESET_ALL)
    print(color_code + text)


def read_bed(file_path: str) -> pd.DataFrame:
    bed_cols = ['chromosome', 'start', 'end', 'name']
    bed = pd.read_csv(file_path, sep='\t', header=None, names=bed_cols)
    return bed

def write_bed(data_dict: dict, file_path: str) -> None:
    with open(file_path, 'w') as f:
        for chrom, array in data_dict.items():
            array = np.array(array)
            n = len(array)
            if n == 0:
                continue  # No data for this chromosome
            edges = np.where(array[1:] != array[:-1])[0] + 1
            if array[0] == 1:
                starts = [0] + edges[1::2].tolist()
                ends = edges[::2].tolist() + [n]
            else:
                starts = edges[::2].tolist()
                ends = edges[1::2].tolist()
                if array[-1] == 1:
                    ends.append(n)
            for s, e in zip(starts, ends):
                f.write(f"{chrom}\t{s}\t{e}\n")


def read_fasta(file_path, first_n=99999):
    sequences = {
        record.id.split()[0] : record.seq
        for _, record in tqdm(
            zip(range(first_n), SeqIO.parse(file_path, 'fasta')),
            desc="Reading FASTA file",
            leave=False
        )
    }
    return sequences

def sequences_to_arrays(sequences, mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}):
    array_dict = {}
    for key, seq in tqdm(sequences.items(), desc="Transcripting strings to np.arrays", leave=False):
        array_dict[key] =  np.array(
            [mapping[char] for char in seq],
            dtype=np.int8
        )
    return array_dict

def _process_sequence(item):
    mapping={'A': 0, 'C': 1, 'G': 2, 'T': 3}
    key, seq = item
    try:
        array = np.array([mapping[char] for char in seq], dtype=np.int8)
        return key, array
    except KeyError as e:
        print(f"Warning: Invalid character '{e.args[0]}' in sequence '{key}'. Skipping this sequence.")
        return key, None

def sequences_to_arrays_parallel(sequences, num_processes=1):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(_process_sequence, sequences.items())
    
    array_dict = {}
    for key, array in results:
        if array is not None:
            array_dict[key] = array
    return array_dict

def map_regions(fasta_dict, bed_df):
    """
    Maps each chromosome to a tuple of two lists: CpG regions and non-CpG regions.

    Parameters:
    - fasta_dict (dict): A dictionary where keys are chromosome IDs and values are sequences as strings.
    - bed_df (pd.DataFrame): A DataFrame containing BED regions with columns 'chromosome', 'start', 'end', 'name'.

    Returns:
    - dict: A dictionary where keys are chromosome IDs and values are tuples of two lists:
            (list of CpG regions, list of non-CpG regions).
    """
    region_dict = {}
    bed_chroms = bed_df['chromosome'].unique()
    
    # Process chromosomes present in the BED file
    for chrom in tqdm(bed_chroms, desc="Mapping chromosomes to genome regions", leave=False):
        regions = bed_df[bed_df['chromosome'] == chrom].sort_values('start')
        chrom_seq = fasta_dict.get(chrom, '')
        seq_len = len(chrom_seq)
        cpg_regions = []
        non_cpg_regions = []
        current_pos = 0
        for index, row in regions.iterrows():
            start = row['start']
            end = row['end']
            if current_pos < start:
                non_cpg_seq = chrom_seq[current_pos:start]
                non_cpg_regions.append(non_cpg_seq)
            cpg_seq = chrom_seq[start:end]
            cpg_regions.append(cpg_seq)
            current_pos = end
        if current_pos < seq_len:
            non_cpg_seq = chrom_seq[current_pos:seq_len]
            non_cpg_regions.append(non_cpg_seq)
        region_dict[chrom] = (cpg_regions, non_cpg_regions)
    
    # Handle chromosomes present in fasta_dict but not in bed_df
    all_chroms = set(fasta_dict.keys())
    bed_chroms_set = set(bed_chroms)
    non_bed_chroms = all_chroms - bed_chroms_set
    for chrom in non_bed_chroms:
        chrom_seq = fasta_dict[chrom]
        seq_len = len(chrom_seq)
        cpg_regions = []
        non_cpg_regions = [chrom_seq]
        region_dict[chrom] = (cpg_regions, non_cpg_regions)
    
    return region_dict


def mask_regions(fasta_dict, bed_df):
    """
    Creates a mask for each chromosome where positions mentioned in the BED file are marked with 1s and others with 0s.

    Parameters:
    - fasta_dict (dict): A dictionary where keys are chromosome IDs and values are sequences as strings.
    - bed_df (pd.DataFrame): A DataFrame containing BED regions with columns 'chromosome', 'start', 'end', 'name'.

    Returns:
    - dict: A dictionary where keys are chromosome IDs and values are numpy arrays representing the mask.
    """
    mask_dict = {}
    bed_chroms = bed_df['chromosome'].unique()

    # Process chromosomes present in the BED file
    for chrom in tqdm(bed_chroms, desc="Creating masks for chromosomes", leave=False):
        regions = bed_df[bed_df['chromosome'] == chrom].sort_values('start')
        chrom_seq = fasta_dict.get(chrom, '')
        if not isinstance(chrom_seq, np.ndarray):
            continue

        seq_len = len(chrom_seq)
        mask = np.zeros(seq_len, dtype=int)

        for index, row in regions.iterrows():
            start = row['start']
            end = row['end']
            mask[start:end] = 1

        mask_dict[chrom] = mask

    # Handle chromosomes present in fasta_dict but not in bed_df
    all_chroms = set(fasta_dict.keys())
    bed_chroms_set = set(bed_chroms)
    non_bed_chroms = all_chroms - bed_chroms_set
    for chrom in non_bed_chroms:
        chrom_seq = fasta_dict[chrom]
        seq_len = len(chrom_seq)
        mask = np.zeros(seq_len, dtype=int)
        mask_dict[chrom] = mask

    return mask_dict
