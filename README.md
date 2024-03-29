
# Bloomz Model Server

This repo provides the code for serving the original bloomz-3b model in production for german snippet generation.
It can be used to spin up a simple HTTP server, that handles translation and snippet generation.

> _**NOTE:** There is an even faster server for this model, built with the Potassium framework. Please use the Potassium Server for optimal performance._
> 
> [![GitHub](https://badgen.net/badge/icon/bloomz%20potassium%20model%20server/orange?icon=github&label)](https://github.com/snipaid-nlg/bloomz-model-server-v2)

## Quickstart:

Curious to get your hand on a bloomz server capable of german snippet generation?

You can check it out with docker:

1. Run `docker build -t bloomz-model-server . && docker run -it bloomz-model-server` to build and run the docker container.

Or you can check it out manually:

1. Run `pip3 install -r requirements.txt` to download dependencies.
2. Run `python3 server.py` to start the server.
3. Run `python3 test.py` in a different terminal session to test against it.

*Note: Model requires a GPU with ~ 15GB memory for generation!*

## Overview:

1. `app.py` contains the code to load and run the model for inference.
2. You can run a simple test with `test.py`!

if deploying using Docker:

3. `download.py` is a script to download our finetuned model weights at build time.

## Production:

This repo provides you with a functioning http server for our finetuned gptj-title-teaser-10k model. You can use it as is, or package it up with our provided `Dockerfile` and deploy it to your favorite container hosting provider!

We are currently running this code on [Banana](https://banana.dev), where you can get 1 hour of model hosting for free. Feel free to choose a different hosting provider. In the following section we provide instructions for deployment with Banana.

# 🍌

# To deploy to Banana Serverless:

- Fork this repo
- Log in to the [Banana App](https://app.banana.dev)
- Select your forked repo for deploy

It'll then be built from the dockerfile, optimized, then deployed on Banana Serverless GPU cluster.  
You can monitor buildtime and runtime logs by clicking the logs button in the model view on the [Banana Dashboard](https://app.banana.dev).

## Demo Integration:

When build and optimization finished successfully you will find your credentials printed in the build logs.

```
Your model was updated and is now deployed!
It is runnable with the same credentials:

API_KEY=Your-Personal-Api-Key
MODEL_KEY=Your-Personal-Model-Key
```

You need these keys to hook up the web app with the model.  
To setup the frontend follow the instructions in the [web-app](https://github.com/snipaid-nlg/web-app) repository.
