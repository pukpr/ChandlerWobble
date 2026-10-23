 #!/bin/bash
 
 source ./plugin_path

# (1) Run test_cw in background
./test_cw &
FIRST_PID=$!

# (2) Run gazebo executable in foreground
gz sim -r earth_centric_wobble.sdf

kill $FIRST_PID 2>/dev/null


