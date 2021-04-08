def create_simpson(output_path, output_name, sw, np, spin_rate, proton_frequency, crystal_file='rep2000', gamma_angles=45, lb=1000, lb_ratio=1.0, cs_iso=0.0, csa=0.0, csa_eta=0.0, alpha=0.0, beta=0.0, gamma=0.0):  # Write simpson input files
    """This generates a custom Simpson inputfile, which can be used from the terminal with: 'simpson <output_name>'

    Args:
        output_path (str): Path to save the generated file
        output_name (str): Name of generated inputfile
        sw (float): Spectral width in Hz
        np (float): Number of FID points
        spin_rate (float): MAS rate in Hz
        proton_frequency (float): Proton frequency in Hz
        crystal_file (str, optional): Crystal file chosen from e.g. rep20 rep2000 rep256 rep678 zcw232 zcw4180 zcw28656. Defaults to 'rep2000'.
        gamma_angles (int, optional): Gamma angles, should be at least sqrt(crystal_file orientations) Defaults to 45.
        lb (int, optional): Linebroadening in Hz. Defaults to 1000.
        lb_ratio (float, optional): Ratio between Gauss/Lorentz linebroadening. Defaults to 1.0.
    """
    import os
    import textwrap
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    simpson_input = """\
    spinsys ⁍
        channels 1H
        nuclei 1H
        shift 1 {cs_iso}p {csa}p {csa_eta} {alpha} {beta} {gamma}
    ⁌

    par ⁍
        spin_rate        {spin_rate}
        proton_frequency {proton_frequency}
        start_operator   Inx
        detect_operator  Inp
        method           direct
        crystal_file     {crystal_file}
        gamma_angles     {gamma_angles}
        sw               {sw}
        variable tsw     1e6/sw
        verbose          0000
        np               {np}
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

        faddlb $f {lb} {lb_ratio}
        fzerofill $f $par(si)
        fft $f

        fsave $f {output_name}.spe
        fsave $f {output_name}.xy -xreim
    ⁌
    """
    simpson_variables = {
        "cs_iso":cs_iso,
        "csa":csa,
        "csa_eta":csa_eta,
        "alpha":alpha,
        "beta":beta,
        "gamma":gamma,
        "spin_rate":spin_rate,
        "proton_frequency":proton_frequency,
        "crystal_file":crystal_file,
        "gamma_angles":gamma_angles,
        "sw":sw,
        "np":np,
        "lb":lb,
        "lb_ratio":lb_ratio,
        "output_name":output_name
    }

    with  open(output_path + output_name,'w') as myfile:
        myfile.write(textwrap.dedent(simpson_input.format(**simpson_variables).replace("⁍", "{").replace("⁌", "}")))


def run_simpson(input_file, working_dir):
    from subprocess import run
    import os

    os.chdir(working_dir)

    run(['simpson', input_file])
