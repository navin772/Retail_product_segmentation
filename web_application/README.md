# Retail Web Interface

This `/web_application` directory contains the necessary files for creating the Web UI and deploying it via Docker containers or orchestrating them through Kubernetes.

## Directory Structure

```
.
├── helm_chart_retail
│   ├── Chart.yaml
│   ├── retail-0.x.0.tgz
│   ├── templates
│   └── values.yaml
├── YAML_files
│   ├── deployment.yaml
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   └── service.yaml
├── requirements.txt
├── Dockerfile
├── main.py
├── heatmap.py
├── fashion.py
├── last.pt
```

Contents of each directory:
- `helm_chart_retail`: Contains the Helm chart for deploying the Web UI on Kubernetes.
- `YAML_files`: Contains the kubernetes manifest files for deploying on Kubernetes clusters with the help of `kustomize`.
- `requirements.txt`: Contains the Python dependencies for the Web UI.
- `Dockerfile`: Contains the Dockerfile for building the Docker image.
- `main.py`: The main Python script
- `heatmap.py`: Contains the code for generating the heatmap.
- `fashion.py`: Contains the code for generating the fashion recommendations.
- `last.pt`: Contains the weights for the model. 

## Deploying/using the Web application
You can use/deploy the Web application in many ways - locally, as a docker container, or on a Kubernetes cluster. 

### Locally

1. Clone the repository and change into the `web_application` directory.

    - `git clone https://github.com/navin772/Retail_product_segmentation.git`
    - `cd web_application`

2. Install the dependencies in your main environment or preferrably in a virtual environment. 

    - `pip install -r requirements.txt`

3. Make sure to have [docker](https://www.docker.com/) and [docker-compose](https://docs.docker.com/compose/install/) installed on your system. Download the [milvus standalone](https://milvus.io/docs/v2.2.x/install_standalone-docker.md0) docker compose file and run it, wait for all the containers to start.

    - `wget https://github.com/milvus-io/milvus/releases/download/v2.2.14/milvus-standalone-docker-compose.yml -O docker-compose.yml`
    - `docker-compose up -d`

4. Prepare and populate the milvus database using the following steps:
    - Download the data to insert  `curl -LO https://github.com/navin772/Retail_product_segmentation/releases/download/v1.0/insert_data_milvus.zip`.
    - Unzip the zip file `unzip -q insert_data_milvus.zip`.
    - Change into the directory `cd insert_data_milvus` and run the script `python insert_data.py`.
    - This will create a collection named `clothing` and insert the data into it.

5. To run the application locally,comment out lines 11-13 in `fashion.py`. Also be sure to comment the lines 6-8 which are for kubernetes deployment purposes. Now run the following command to start the application.

    - `streamlit run main.py` 

6. Access the web application at `http://localhost:8501/`.


### Docker container deployment

1. Follow steps 3 and 4 from the previous section - [**Locally**](https://github.com/navin772/Retail_product_segmentation/tree/docker-build/web_application#locally) to setup the Milvus database.
2. Pull the docker image from dockerhub.

    - `docker pull navin772/retail:deploy_container`

3. Run the docker image. The `--network="host"` flag is used to connect the docker container to the host network. This is done so that the container can access the milvus server running on the host machine.

    - `docker run --network="host" navin772/retail:deploy_container`

4. Access the web application on the *Network URL* provided by the docker container - http://192.168.0.110:8501/ in this case.

### Kubernetes cluster deployment

1. Follow the milvus kubernetes deployment guide to setup Milvus on a Kubernetes cluster. 

    - https://milvus.io/docs/install_standalone-helm.md

2. Port-forward the Milvus service to access it from the host machine.

    - `kubectl port-forward service/milvus 19530:19530`

3. Follow the step 4 from the section - [**Locally**](https://github.com/navin772/Retail_product_segmentation/tree/docker-build/web_application#locally) to populate the Milvus database.

4. Change into the `web_application` directory and run the following command to deploy the application on the Kubernetes cluster using kustomize.

    - `kubectl apply -k YAML_files/`

5. Access the web application at `NodeIP:NodePort` where `NodeIP` is the IP address of the node on which the application is deployed and `NodePort` is the port number on which the application is exposed. 

    - `kubectl get nodes -o wide` - To get the IP address of the node.
    - `kubectl get svc -n retail` - To get the port number on which the application is exposed.