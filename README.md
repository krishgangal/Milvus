# Milvus
Using Sentence Transformers and Milvus to generate embeddings on job postings and recognize duplicates.

Download jobpostings.csv and add all files to one folder. 

In command line, navigate to that folder and run "docker compose up --build" 

Docker will build four containers; the krishgnagal_script container will run the python script and the output will be in the container's logs. 

These are the rows of the sheet that contain duplicate listings.

Notes: 
I only used the first 100 lines of jobpostings.csv as my computing power is limited.
Only tested on Mac.
