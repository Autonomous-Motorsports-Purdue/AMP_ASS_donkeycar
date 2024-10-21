#!/usr/bin/env bash
PORT=6000
echo "Staring TCP Echo server on port $PORT..."
socat -v tcp-l:$PORT,fork exec:'/bin/cat'
