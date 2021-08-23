web: pip install 'dvc[s3]'
web: git init
web: dvc pull
web: uvicorn web.app:app --host=0.0.0.0 --port=${PORT:-5000}