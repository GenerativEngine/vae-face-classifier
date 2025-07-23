@echo off
SETLOCAL

REM --- Configuration ---
SET SCRIPT_NAME=autoencoder_plot_face_recognition.py  
SET APP_NAME=face-recognition-api
SET DOCKER_USERNAME=your-docker-username-here REM <<< IMPORTANT: REPLACE THIS WITH YOUR DOCKER HUB USERNAME OR LEAVE BLANK FOR LOCAL IMAGE
SET KUBERNETES_SERVICE_TYPE=NodePort REM <<< IMPORTANT: Change to LoadBalancer if deploying to cloud K8s

REM --- Step 1: Train Model & Save ---  
echo.
echo --- Step 1: Running training script to generate models ---  
python %SCRIPT_NAME%
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Training script failed. Exiting.
    GOTO :EOF    
)

REM --- Step 2: Create models directory and move saved models ---
echo.  
echo --- Step 2: Organizing models for Docker build ---  
IF NOT EXIST models MD models
MOVE encoder_model.h5 models\
MOVE svm_classifier.pkl models\
MOVE target_names.pkl models\ 
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to move models. Ensure they were generated. Exiting.
    GOTO :EOF 
)

REM --- Step 3: Build Docker Image ---  
echo.
echo --- Step 3: Building Docker image ---
docker build -t %DOCKER_USERNAME%/%APP_NAME%:latest .  
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Docker image build failed. Exiting.
    GOTO :EOF  
)

REM --- Step 4: Push Docker Image (Optional - Uncomment if deploying to cloud K8s) ---
REM echo.
REM echo --- Step 4: Pushing Docker image to registry --- 
REM docker push %DOCKER_USERNAME%/%APP_NAME%:latest
REM IF %ERRORLEVEL% NEQ 0 (
REM     echo Error: Docker image push failed. Exiting.
REM     GOTO :EOF
REM )
 
REM --- Step 5: Deploy to Kubernetes --- 
echo.
echo --- Step 5: Deploying to Kubernetes ---
REM Ensure the image name in k8s-deployment.yaml matches %DOCKER_USERNAME%/%APP_NAME%:latest
kubectl apply -f k8s-deployment.yaml 
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Kubernetes deployment failed. Exiting.
    GOTO :EOF 
)

REM Ensure the service type in k8s-service.yaml matches %KUBERNETES_SERVICE_TYPE%
kubectl apply -f k8s-service.yaml
IF %ERRORLEVEL% NEQ 0 (
    echo Error: Kubernetes service creation failed. Exiting. 
    GOTO :EOF
)

echo.
echo --- Deployment Complete! ---
echo.
echo To access your API: 
IF "%KUBERNETES_SERVICE_TYPE%"=="NodePort" (
    echo For Minikube (NodePort):
    minikube service %APP_NAME%-service --url 
) ELSE (
    echo For Cloud Kubernetes (LoadBalancer):
    echo Run 'kubectl get service %APP_NAME%-service' to get the external IP.
) 

ENDLOCAL
