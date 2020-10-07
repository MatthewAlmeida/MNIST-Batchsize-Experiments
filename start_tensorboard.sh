#!/bin/bash

# Pass command to bash shell inside container so environment variables
# are evaluated inside the container.
#
# Requires MBEModeling container to be running.
docker exec -it MBEModeling bash -c 'tensorboard --logdir="/$LOG_DIR" --port=$TB_PORT --host=0.0.0.0'