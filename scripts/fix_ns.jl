using FastaIO
using DataFrames
using CSV
using ArgParse

"""Checks if bed regions do not overlap"""
function is_valid_bed(bed_df::DataFrame)::Bool
	if isempty(bed_df)
		@warn "BED file dataframe is empty"
		return true
	end
	sort!(bed_df, [:chrom, :chromStart])
	chroms = unique(bed_df.chrom)
	for chr in chroms
		chrom_df = bed_df[bed_df.chrom .== chr, :]
		prev_end = 0
		@simd for row in eachrow(chrom_df)
			if row.chromEnd <= row.chromStart < prev_end
				return false
			end
			prev_end = row.chromEnd
		end
	end
	return true
end

"""Read bed file with columns `chrom`, `chromStart`, `chromEnd` from disc"""
function parse_bed_file(filepath::String)::DataFrame
    df = CSV.read(filepath, DataFrame, header=false)
    bed_columns = [:chrom, :chromStart, :chromEnd, :name]
    rename!(df, bed_columns)
	return_df = df[!, 1:3]
	@assert is_valid_bed(return_df)
    return return_df
end

function save_bed_to_disk(filename::String, bed_df::DataFrame)
    open(filename, "w") do io
        for row in eachrow(bed_df)
            row_data = [string(row[coln]) for coln in names(bed_df)]
            println(io, join(row_data, "\t"))
        end
    end
end

"""Returns all bed regions as `Dict{String, Vector{String}}`"""
function extract_subsequences(bed_df::DataFrame, seq_tuples::Vector{Tuple{String, String}})::Dict{String, Vector{String}}
	subsequences_dict = Dict{String, Vector{String}}()
    for (header, sequence) in seq_tuples
		chrom = first(split(header))
        chrom_df = bed_df[bed_df.chrom .== chrom, :]
        subsequences = String[]
        for row in eachrow(chrom_df)
			start = row.chromStart + 1
            stop = row.chromEnd
            subsequence = sequence[start:stop]
            push!(subsequences, subsequence)
        end
        if !isempty(subsequences)
            subsequences_dict[chrom] = subsequences
        end
    end

    return subsequences_dict
end

"""Removes any bed regions with `'N'` characters. Filters out all `'N'` characters from between bed regions and shifts remaining bed entries accordingly"""
function filter_ns(bed_df::DataFrame, seq_tuples::Vector{Tuple{String, String}})
	
	"""Removes bed entries, with `'N'` characters"""
	function purge_cpg_with_ns(bed_df::DataFrame, seq_tuples::Vector{Tuple{String, String}})::DataFrame
		@assert is_valid_bed(bed_df)
		new_df = DataFrame(chrom=String[], chromStart=Int[], chromEnd=Int[])

	    for (header, sequence) in seq_tuples
			chrom_i = first(split(header))
	        chrom_df = bed_df[bed_df.chrom .== chrom_i, :]
	        for row in eachrow(chrom_df)
				start = row.chromStart + 1
	            stop = row.chromEnd
	            subsequence = sequence[start:stop]
	            if !occursin('N', subsequence)
	                push!(new_df, (chrom=chrom_i, chromStart=row.chromStart, chromEnd=row.chromEnd))
	            end
	        end
	    end
		@assert is_valid_bed(new_df)
	    return new_df
	end

	bed_df = purge_cpg_with_ns(bed_df, seq_tuples)
	# double check
	@assert all.(
        !=('N'),
        vcat(values(extract_subsequences(bed_df, seq_tuples))...)
    ) |> all "Bed regions must not contain 'N' characters!"
	
    chrom_to_seq = Dict{String, String}()
    for (header, seq) in seq_tuples
        chrom_id = split(header, ' ')[1]
        chrom_to_seq[chrom_id] = seq
    end

    filtered_bed_regions = @NamedTuple{chrom::String, chromStart::Int64, chromEnd::Int64}[]
    filtered_seq_list = Tuple{String, String}[]

    for (header, seq) in seq_tuples
        chrom_id = split(header, ' ')[1]
        filtered_seq = replace(seq, "N" => "")
        n_counts = zeros(Int, length(seq) + 1)
        n_count = 0
        for i in 1:length(seq)
            if seq[i] == 'N'
                n_count += 1
            end
            n_counts[i + 1] = n_count
        end
        chrom_bed = bed_df[bed_df.chrom .== chrom_id, :]
        for row in eachrow(chrom_bed)
            original_start = row.chromStart
            original_end = row.chromEnd
            new_start = original_start - n_counts[original_start + 1]
            new_end = original_end - n_counts[original_end]
            push!(filtered_bed_regions, (chrom=row.chrom, chromStart=new_start, chromEnd=new_end))
        end
        push!(filtered_seq_list, (header, filtered_seq))
    end

    filtered_bed_df = DataFrame(
        chrom = [row.chrom for row in filtered_bed_regions],
        chromStart = [row.chromStart for row in filtered_bed_regions],
        chromEnd = [row.chromEnd for row in filtered_bed_regions]
    )

    return filtered_bed_df, filtered_seq_list
end


function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--fasta_in"
        help = "Input FASTA file"
        required = true
        
        "--bed_in"
        help = "Input BED file"
        required = true

        "--fasta_out"
        help = "Input FASTA file"
        required = true
        
        "--bed_out"
        help = "Input BED file"
        required = true
    end
    args = parse_args(s)

    genome = [(header, uppercase(seq)) for (header, seq) in FastaReader(args["fasta_in"]) if startswith(header, "NC_054")]
    bed_df = parse_bed_file(args["bed_in"])
    filtered_bed, filtered_genome = filter_ns(bed_df, genome)
    save_bed_to_disk(args["bed_out"], filtered_bed)

    println("Calculations are done, now writing file to disc...")
    writefasta(args["fasta_out"], filtered_genome; check_description=false)
end

if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
end



