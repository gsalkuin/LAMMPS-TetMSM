#!/usr/bin/env bash
mkdir -p data
for d in nu-*/; do
	cp "$d"*-nu-*.txt data/ 2>/dev/null
done