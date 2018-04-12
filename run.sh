#!/usr/bin/env bash
docker run -it \
	-p 5901:5901 \
	-p 6901:6901 \
	-v ~/Dropbox/Projects/gym-minigrid:/headless/gym-minigrid \
	gym-minigrid \
	bash