#!/usr/bin/env bash

set -e 

DATAFOLDER=${1:-/data/$(whoami)/}
MODELFOLDER=${2:-/data/$(whoami)/models}

DATAFOLDER=${DATAFOLDER%/}  # remove trailing slash
MODELFOLDER=${MODELFOLDER%/}  # remove trailing slash

xhost local:root

# pull docker image
docker pull roshambo919/scenegraph:benchmark

# check if the data folders are populated before continuing
if [ -z "$(ls -A $DATAFOLDER)" ]; then
   echo "\nERROR: nothing found in the data directory at $DATAFOLDER. Ensure you have downloaded the data before starting container"
   exit
else
   echo "Data directory not empty...assuming it is populated with data!"
fi
if [ -z "$(ls -A $MODELFOLDER)" ]; then
   echo "\nERROR: nothing found in the model directory at $MODELFOLDER. Ensure you have downloaded models before starting container"
   exit
else
   echo "Model directory not empty...assuming it is populated with models!"
fi


# function to start docker
start_docker () {
    DF=$1
    MF=$2
    echo "Starting fresh docker container"
    docker run \
      --name sgg-benchmark \
      --privileged \
      --runtime=nvidia \
      --gpus 'all,"capabilities=graphics,utility,display,video,compute"' \
      --mount type=bind,src="$DATAFOLDER",target=/data \
      --mount type=bind,src="$MODELFOLDER",target=/models \
      -p 8080:8080 \
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
start_docker $DATAFOLDER $MODELFOLDER


exit 0