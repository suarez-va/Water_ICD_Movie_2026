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

set joblist  $env(VMDMOVIE_JOBLIST)
set settings $env(VMDMOVIE_SETTINGS)
set width    $env(VMDMOVIE_WIDTH)
set height   $env(VMDMOVIE_HEIGHT)
set renderer $env(VMDMOVIE_RENDERER)

# Tell vmd_settings.tcl not to apply itself on source; we drive it per frame.
set NO_AUTO_APPLY 1
source $settings

# Load-bearing, and it must come before any render: the camera written into the
# scene is derived from the display's aspect ratio, so the display has to be at
# the target size or the external renderer produces a stretched image.
display resize $width $height
setup_scene

set fp [open $joblist r]
set lines [split [read $fp] "\n"]
close $fp

set n 0
foreach line $lines {
    if {[string trim $line] eq ""} { continue }
    set parts [split $line "\t"]
    set cube [lindex $parts 0]
    set out  [lindex $parts 1]

    set molid [mol new $cube type cube waitfor all]
    setup_frame $molid
    setup_view  $molid
    # "external" writes a Tachyon scene file, which render_movie.py then patches
    # (to lower the AO/antialiasing sample counts, the dominant render cost) and
    # feeds to the standalone tachyon binary.  TachyonInternal hardcodes 12/12
    # and offers no way to change them.
    if {$renderer eq "external"} {
        render Tachyon $out
    } else {
        render TachyonInternal $out
    }
    mol delete $molid
    incr n
    puts "FRAME_DONE $out"
    flush stdout
}

puts "RENDER_LOOP_DONE $n"
quit
