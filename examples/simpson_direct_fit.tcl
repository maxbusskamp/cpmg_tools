spinsys {
  channels 207Pb
  nuclei 207Pb 207Pb
  shift 1 -218p 0p 0.0 0 0 0
  shift 2 128p 0p 0.0 0 0 0
}

par {
    spin_rate        12500
    proton_frequency 500e6
    start_operator   Inx
    detect_operator  Inp
    method           direct
    crystal_file     rep30
    gamma_angles     7
    sw               2.5e6
    variable tsw     1e6/sw
    verbose          0000
    np               4096*4
    variable si      np*2
}


proc pulseq {} {
    global par
    acq_block {
	    delay $par(tsw)
    }
}

proc main {} {
    global par

    lappend ::auto_path /usr/local/tcl
    lappend ::auto_path ./opt 1.1
    if {![namespace exists opt]} {
        package require opt 1.1
        package require simpson-utils
        namespace import simpson-utils::*
    }
    set par(verb) 1

    # load experimental spectrum
    set par(exp) [fload 207Pb_PbZrO3_MAS_WCPMG_1.spe]

    # declare fit function (no change required)
    opt::function rms

    #parameter name start_value, inc., lower_limit, upper_limit (change)
    opt::newpar csa1 -200 1 -700 -100
    opt::newpar eta1 0.1 0.1 0 0.5
    opt::newpar csa2 -1100 1 -1200 -700
    opt::newpar eta2 0.25 0.1 0 0.5


    # opt::scan cs1
    # opt::scan csa1
    # opt::scan eta1
    # opt::scan cs2
    # opt::scan csa2
    # opt::scan eta2
    # opt::scan lb

    # start optimization of function "rms" with variables declared above
    opt::minimize 1.0e-5


    rms 1
}


proc rms {{save 0}} {
    global par

    set f [fsimpson [list \
                    [list shift_1_aniso ${opt::csa1}p] \
                    [list shift_1_eta $opt::eta1] \
                    [list shift_2_aniso ${opt::csa2}p] \
                    [list shift_2_eta $opt::eta2] \
                    ]]

    faddlb $f 2300 1.0
    fzerofill $f $par(si)
    fft $f

    # normalize and scale correctly
    fautoscale $f $par(exp) -re

    # save test spectrum
    fsave $f $par(name)_test.spe

    # calculate RMS, compare and show results
    set rms [frms $f $par(exp) -re ]
    if {$save == 1} {
        puts "cs_iso1       cs_iso2       csa1      csa2      eta1       eta2       lb       rms"
        puts [format " \[%s\] %10.3f %10.3f %10.3f %10.3f %10.3f" \
        FINAL $opt::csa1 $opt::csa2 $opt::eta1 $opt::eta2 $rms]
        fsave $f $par(name)_final.spe
    }
    flush stdout
    funload $f
    return $rms
}
