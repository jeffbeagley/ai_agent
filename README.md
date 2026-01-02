# ai_agent

Template for deploying langchain based projects. 

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

Use the provided `run.sh` script to start the application, replacing <user_input> with your prompt.

```bash
./run.sh "<user_input>"
```
