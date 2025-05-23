#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# copy requirements
cp requirements.txt submit/requirements.txt

# submit team.txt
printf "Kavya Varghese,kavyav\nMax Thompson,thompm22" > submit/team.txt

# train model
python3 src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python3 src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
