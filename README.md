
# Real-time Face Recognition System

A distributed system for real-time face detection and recognition, designed for scalability and performance. The project consists of two components: a web server for API and monitoring, and a backend camera for face detection. Built to demonstrate full-stack skills in backend development, API design, and system architecture.

## Tech Stack

-   **Backend**: FastAPI, Python, OpenCV, YOLO, InsightFace
-   **Database**: Redis
-   **DevOps**: Nginx, RabbitMQ, Docker, Git, GitHub
-   **Hardware**: Raspberry Pi
-   **Tools**: Postman, VS Code

## Features

-   Real-time face detection using YOLO and OpenCV on Raspberry Pi
-   Facial recognition with InsightFace, storing embeddings in Redis
-   Scalable architecture with Nginx load balancing and RabbitMQ for workload management
-   Monitoring system with event logging (timestamp, identity, location)
-   REST API for accessing recognition data and logs

## Installation

Follow these steps to set up and run the project locally.

### Prerequisites
-   Docker (optional for containerization)
-   Raspberry Pi (for camera component)

### Web Server Setup

1.  Clone the web server repository:
    
    ```bash
    git clone https://github.com/Putthakun/webserver-recognition-SY.git
    cd webserver-recognition-SY
    ```
    
2.  Create docker network:
    
    ```bash
    docker network create SY_network
    ```
        
3.  Run the docker:
	   ```bash
	docker compose up --build
	   ```

### Backend Camera Setup

1.  Clone the backend camera repository:
    
    ```bash
    git clone https://github.com/Putthakun/backend-camera-SY.git
    cd backend-camera-SY
    
    ```
    
2.  Run the docker:
    
    ```bash
    docker compose up --build
    ```
    
3.  Connect Raspberry Pi camera and update config:
    
    

## Usage

```
swagger webserver 1
http://0.0.0.0:8002/docs#/

swagger webserver 2
http://0.0.0.0:8003/docs#/

swagger webserver 3
http://0.0.0.0:8003/docs#/


Rabbit MQ UI
http://0.0.0.0:15672


grafana
http://0.0.0.0:3000
    - username: admin
    - password: admin123
```

## Links

-   Web Server Repository: [github.com/Putthakun/webserver-recognition-SY](https://github.com/Putthakun/webserver-recognition-SY)
-   Backend Camera Repository: [github.com/Putthakun/backend-camera-SY]

## Contact

-   Email: [putthakun01@gmail.com](mailto:putthakun01@gmail.com)
-   GitHub: [github.com/Putthakun](https://github.com/Putthakun)
