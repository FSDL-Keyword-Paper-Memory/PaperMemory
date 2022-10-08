#!/bin/sh

docker build -t keyword-extractor . && docker run -d -p 2137:5000 --name keyword-extractor keyword-extractor
