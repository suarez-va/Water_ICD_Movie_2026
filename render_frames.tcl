# ---------------------------------------------------------------------------
# Render loop driven by render_movie.py.  Not meant to be edited for looks --
# put visual changes in vmd_settings.tcl instead.
#
# This script is fed to VMD on stdin, NOT with -e:
#
#   VMDMOVIE_JOBLIST=... vmd -dispdev text < render_frames.tcl
#
# The distinction matters.  A script given with `-e` runs during VMD's startup,
# before the display device exists, so `display resize` throws -- and VMD then
# silently abandons the rest of the file with no error and exit status 0.  On
# stdin the script runs after initialisation and everything works.  Parameters
# arrive through the environment because `-args` is only wired up for `-e`.
# ---------------------------------------------------------------------------

# Stage timings, in ms, for render_movie.py --profile.  Always printed; the
# driver ignores them unless asked.
set t_script [clock milliseconds]

set joblist  $env(VMDMOVIE_JOBLIST)
set settings $env(VMDMOVIE_SETTINGS)
set width    $env(VMDMOVIE_WIDTH)
set height   $env(VMDMOVIE_HEIGHT)
set renderer $env(VMDMOVIE_RENDERER)
set ao_samples $env(VMDMOVIE_AO_SAMPLES)
set aa_samples $env(VMDMOVIE_AA_SAMPLES)

# Tell vmd_settings.tcl not to apply itself on source; we drive it per frame.
set NO_AUTO_APPLY 1
source $settings

# Load-bearing, and it must come before any render: the camera written into the
# scene is derived from the display's aspect ratio, so the display has to be at
# the target size or the external renderer produces a stretched image.
display resize $width $height
setup_scene

# The GPU renderer takes its sample counts as VMD settings rather than from a
# scene file, so they are set once here instead of patched per frame.
if {$renderer eq "optix"} {
    render aasamples TachyonLOptiXInternal $aa_samples
    render aosamples TachyonLOptiXInternal $ao_samples
}

puts "TIMING startup [expr {[clock milliseconds] - $t_script}]"

set fp [open $joblist r]
set lines [split [read $fp] "\n"]
close $fp

set n 0
foreach line $lines {
    if {[string trim $line] eq ""} { continue }
    set parts [split $line "\t"]
    set cube [lindex $parts 0]
    set out  [lindex $parts 1]

    set t0 [clock milliseconds]
    set molid [mol new $cube type cube waitfor all]
    set t1 [clock milliseconds]
    setup_frame $molid
    setup_view  $molid
    set t2 [clock milliseconds]
    # "external" writes a Tachyon scene file, which render_movie.py then patches
    # (to lower the AO/antialiasing sample counts, the dominant render cost) and
    # feeds to the standalone tachyon binary.  TachyonInternal hardcodes 12/12
    # and offers no way to change them.
    # "optix" ray-traces on an NVIDIA GPU; it needs a CUDA/OptiX build of VMD.
    if {$renderer eq "external"} {
        render Tachyon $out
    } elseif {$renderer eq "optix"} {
        render TachyonLOptiXInternal $out
    } else {
        render TachyonInternal $out
    }
    set t3 [clock milliseconds]
    puts "TIMING frame [file rootname [file tail $cube]] load [expr {$t1 - $t0}] setup [expr {$t2 - $t1}] render [expr {$t3 - $t2}]"
    mol delete $molid
    incr n
    puts "FRAME_DONE $out"
    flush stdout
}

puts "RENDER_LOOP_DONE $n"
quit
