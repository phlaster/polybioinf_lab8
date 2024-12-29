using FastaIO, ArgParse, Printf


function main()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--fasta"
        help = "Input FASTA file"
        required = true
    end
    args = parse_args(s)
    
    println("length\t Ns\t%Ns\taccession")
    for (id, seq) in FastaReader(args["fasta"])
        Ns = count('N', uppercase(seq))
        @printf("%.1e  ", length(seq))
        @printf("%d\t", Ns)
        @printf("%.4f", Ns/length(seq))
        println("%\t", id |> split |> first)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    @time main()
end
