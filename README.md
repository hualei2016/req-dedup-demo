This project is used to give a demo of a requirement dedup tool, which uses LLM capability to find duplicated requirements, and provide analysis and merging suggestions.
The project can be deplyed to GCP. The main file is app.py, with front-end files in templates/*.html, and config files: app.yaml and requirements.txt

In order to make it work, you also need to provide two keys:
* In app.yaml, you need to DASHSCOPE_API_KEY to your own key to access TongYi
* You also need to provide a json file containing the gcs (bucket) key, to use object storage of google cloud
