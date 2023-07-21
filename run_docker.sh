#!/usr/bin/env bash

set -e 

MODELFOLDER=${1:-/data/$(whoami)/models}

xhost local:root

# pull docker image
docker pull roshambo919/scenegraph:benchmark

DEST_TORCH_HOME=/workspace/models

# function to start docker
start_docker () {
    echo "Starting fresh docker container"
    docker run \
      --name sgg-benchmark \
      --privileged \
      --runtime=nvidia \
      --gpus 'all,"capabilities=graphics,utility,video,compute"' \
      -p 8080:8080 \
      -e DISPLAY=$DISPLAY \
      -e TORCH_HOME=$DEST_TORCH_HOME \
      --mount type=bind,src="$MODELFOLDER",target="$DEST_TORCH_HOME" \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -it \
      roshambo919/scenegraph:benchmark \
       /bin/bash
}


# Remove if there is existing container
CONT_ID=$(docker ps -aqf "name=^sgg-benchmark")
if [ "$CONT_ID" == "" ];
then
	:
else
	echo "Stopping and removing existing docker container"
	docker stop $CONT_ID
	docker rm $CONT_ID
fi

# Start up a docker container
echo "Starting up docker container"
start_docker


exit 0