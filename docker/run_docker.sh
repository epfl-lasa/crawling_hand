#!/bin/bash

# Default options
DOCKER=crawling_hand_docker
DOCKERFILE=dev.Dockerfile
NAME=crawling_hand
BUILD=false
WORKSPACE=/home/lars/hand_ws
GIT_WORKSPACE=/home/lars/git

# noetic or humble (untested)
ROS_DISTRO=noetic

help()
{
    echo "Usage: run_docker.sh [ -d | --docker <image name> ]
               [ -b | --build <dockerfile name> ] [ -n | --name <docker name> ]
               [ -w | --workspace </workspace/path> ]
               [ -h | --help  ]"
    exit 2
}

SHORT=d:,b:,n:,w:,h
LONG=docker:,build:,name:,workspace:,help

OPTS=$(getopt -a -n run_docker --options $SHORT --longoptions $LONG -- "$@")

echo $OPTS
eval set -- "$OPTS"

while :
do
  case "$1" in
    -d | --docker )
      DOCKER="$2"
      shift 2
      ;;
    -b | --build )
      BUILD="true"
      DOCKERFILE="$2"
      shift 2
      ;;
    -n | --name )
      NAME="$2"
      shift 2
      ;;
    -w | --workspace )
      WORKSPACE="$2"
      shift 2
      ;;
    -h | --help)
      help
      ;;
    --)
      shift;
      break
      ;;
    *)
      echo "Unexpected option: $1"
      help
      ;;
  esac
done

echo "Docker Build ? $BUILD"

if [ "$BUILD" = true ]; then
     docker build -f $DOCKERFILE -t $DOCKER .
fi

XAUTH=/tmp/.docker.xauth

echo "Preparing Xauthority data..."
xauth_list=$(xauth nlist :0 | tail -n 1 | sed -e 's/^..../ffff/')
if [ ! -f $XAUTH ]; then
    if [ -n "$xauth_list" ]; then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

echo "Done."
echo ""
echo "Verifying file contents:"
file $XAUTH
echo "--> It should say \"X11 Xauthority data\"."
echo ""
echo "Permissions:"
ls -FAlh $XAUTH
echo ""
echo "Running docker..."

docker run -it --rm \
    --env="DISPLAY=$DISPLAY" \
    --volume=$WORKSPACE:/root/hand_ws \
    --volume=/home/$USER/data:/root/data \
    --volume=$GIT_WORKSPACE:/root/git \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --privileged \
    --name=$NAME \
    ${DOCKER} \
    bash
# --gpus all \
