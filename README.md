# Commands to replicate the result:

make sure you have installed **Julia v1.11.2** and **Conda** in your PATH
```sh
$ mkdir DATA
$ wget -P DATA/ https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/017/654/675/GCF_017654675.1_Xenopus_laevis_v10.1/GCF_017654675.1_Xenopus_laevis_v10.1_genomic.fna.gz
$ gunzip DATA/GCF_017654675.1_Xenopus_laevis_v10.1_genomic.fna.gz
$ conda env create --file environment.yaml
$ conda activate bioinf_lab8
```
put bed file into `DATA/` dir as `cpg.bed`

Show genome info:
```sh
$ julia --project=. scripts/genome_info.jl \
    --fasta DATA/GCF_017654675.1_Xenopus_laevis_v10.1_genomic.fna
```

Remove degenerate nucleotides (`N`-s):
```sh
$ julia --project=. scripts/fix_ns.jl \
    --fasta_in DATA/GCF_017654675.1_Xenopus_laevis_v10.1_genomic.fna \
    --bed_in cpg.bed \
    --fasta_out DATA/pruned_only_chroms.fa \
    --bed_out DATA/pruned_only_chroms.bed
```

Train model and get predictions
```sh
$ python scripts/prog.py --fasta DATA/pruned_only_chroms.fa \
    --bed_in DATA/pruned_only_chroms.bed \
    --cpu 8 \
    --bed_out out.bed \
    --n_chroms 1 \
    --model_out model.json
```

Analyze the results:
```sh
python scripts/analisys.py \
    --bed_in DATA/pruned_only_chroms.bed \
    --bed_out out.bed \
    --hist_out hist.png \
    --cpu 8 
```