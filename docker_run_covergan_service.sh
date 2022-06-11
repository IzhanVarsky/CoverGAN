#docker run --rm -p 5001:5001 covergan_service:latest
docker run --shm-size=2g --rm -p 5001:5001 -v `pwd`:/scratch --user $(id -u):$(id -g) --workdir=/scratch -e HOME=/scratch covergan_service:latest