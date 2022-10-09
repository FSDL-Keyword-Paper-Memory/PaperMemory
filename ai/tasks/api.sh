#!/bin/sh

docker build --platform linux/amd64 -t keyword-extractor . && docker run --platform linux/amd64 -d -p 2137:5000 --name keyword-extractor keyword-extractor
