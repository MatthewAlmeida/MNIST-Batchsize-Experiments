#!/bin/bash

# Pass command to bash shell inside container so environment variables
# are evaluated inside the container. Somewhat abuses docker's use 
# of root for containers.
#
# Requires MBEModeling container to be running.
docker exec -it MBEModeling bash -c 'rm -r /${LOG_DIR}/*'