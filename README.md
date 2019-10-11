# Pasos para desplegar

## 1) Construir imagen s2i 

    build . seldonio/seldon-core-s2i-python3:0.12 <docker-io-repo>/<docker-tag>:<tag-version>

Ejemplo: 

    s2i build . seldonio/seldon-core-s2i-python3:0.12 gabrielg78/logistic:1.0.0
 
## 2) Subir imagen a docker.io

    docker push <docker-io-repo>/<docker-tag>

## 3) Crear (actualizar) deployment en Kubernetes

En el archivo *model-deployment.yaml* especificar los datos (nombre, versión, etc.) del modelo que se desea desplegar y *aplicarlo* al cluster de k8s respectivo

    kubectl apply -f model-deployment.yaml

## 4) Exponer modelo para acceso público

### 4.1) Obtener nombre del deploy en k8s

    kubectl get deployments

Ejemplo:

    NAME      READY   UP-TO-DATE   AVAILABLE   AGE
    ml-cf1b   0/0     0            0           10d
    scikit-7  1/1     1            1           10d
    ...

### 4.2) exponer puerto

    kubectl expose deployment <deploymen-name> --type=NodePort

### 4.3) Ver puerto obtenido

    kubectl get services

Ejemplo:

    NAME      TYPE ... EXTERNAL-IP  PORT(S) ...
    scikit-75 NodePort ...          9000:30516/TCP ...
    ...

## Probar localmente el modelo desarrollado

### 1) Desplegar modelo
    seldon-core-microservice <model-class> REST
### 2) Invocar api REST contra puerto 5000
    curl -XPOST -H ... http://localhost:5000/predict

## Ver log de deployment (POD)

### 1) obtener nombre del POD

    kubectl get pods

Ejemplo:

    NAME       READY   STATUS    RESTARTS   AGE
    scikit-75  3/3     Running   0          3d18h
    ...

### 2) ver log

    kubectl logs <pod-name> -c <pod-context> -f

*-f para ver el log en "streamming"*