# ai_agent

This is meant to be a template for creating multi-agent workloads in python leveraging langchain/graph, fastapi, openai, and open-webui as a chat interface. The intent
is to get a working scaffolding setup that I can build upon to make personal and professional assistants. The API should be openai compatible to support any client/front end.

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- [LM Studio](https://lmstudio.ai/) installed and running.
	- Download and run the model of your choice, updating the model reference within main.py

## Building the Container

1. Setup project
	```bash
	export PROJECT=ai_agent
	```
2. Clone the repository into folder named based on variable $PROJECT
	```bash
	git clone https://github.com/jeffbeagley/<repo>.git $PROJECT
	cd $PROJECT
	```

3. Build the Docker image:
	```bash
	docker build -t $PROJECT .
	```

## Running the Project

Use the provided `run.sh` script to start the application. 

```bash
./run.sh 
```

Integrate into your interface/client that supports openai spec.
