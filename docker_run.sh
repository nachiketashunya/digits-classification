sudo docker build -t digits:v1 -f docker/Dockerfile .

sudo docker run -v /home/nachiketa/Dropbox/ML-Ops/digits-classification/models:/digits/models digits:v1
