#!/bin/bash
# Forward all arguments to the actual script
"$(dirname "$0")/bsub_script_tiger_normal_reservation.sh" "$@"
