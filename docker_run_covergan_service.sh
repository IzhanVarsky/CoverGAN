docker run --rm -p 5001:5001 covergan_service:latest

docker run --shm-size=2g --rm -v `pwd`:/scratch --user --workdir=/scratch -e HOME=/scratch -p 5001:5001 covergan_service:latest