## setup the python venv
python3 -m venv .venv 

## activate the venv
source .venv/bin/activate

## install dependecies
pip3 install -r requirements.txt

## run the FastAPI app
uvicorn main:app --reload