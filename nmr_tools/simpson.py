def create_simpson(output_path, output_name, spin_rate):  # Write simpson input files

    simpson_input = """\
    spinsys ⁍
        channels 1H
        nuclei 1H
        shift 1 0p 0p 0.0 0 0 0
    ⁌

    par ⁍
        spin_rate        {spin_rate:5.0f}
        proton_frequency 500e6
        start_operator   Inx
        detect_operator  Inp
        method           direct
        crystal_file     rep2000
        gamma_angles     45
        sw               2e6
        variable tsw     1e6/sw
        verbose          0000
        np               4096*2
        variable si      np*2
    ⁌


    proc pulseq ⁍⁌ ⁍
        global par
        acq_block ⁍
            delay $par(tsw)
        ⁌
    ⁌

    proc main ⁍⁌ ⁍
        global par

        set f [fsimpson]

        faddlb $f 1500 1.0
        fzerofill $f $par(si)
        fft $f

        fsave $f output.spe
        fsave $f output.xy -xreim
    ⁌
    """
    simpson_variables = {
        "spin_rate":spin_rate
    }
    with  open(output_path + output_name,'w') as myfile:
        myfile.write(simpson_input.format(**simpson_variables).replace("⁍", "{").replace("⁌", "}"))

def run_simpson(input_file):
    from subprocess import run

    run(['simpson', input_file])
