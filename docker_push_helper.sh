sudo docker build -t digits-base -f docker/Dockerfile.base .
sudo docker build -t digits-test -f docker/Dockerfile.test .

az acr build --image digitsclassification.azurecr.io/digitsbase --registry digitsclassification --resource-group ML-Ops -f docker/Dockerfile.base .