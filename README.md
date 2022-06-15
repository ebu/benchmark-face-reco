# benchmark-face-reco

A framework for benchmarking facial recognition systems

**Creating a virtualenv**

``pipenv install``

**Parallel videos processing**

``ls -1 data/videos/*.mp4 | xargs -n1 -P$(nproc --all) -I {} sh -c 'FILE_NAME=$(basename $1); FILE_STEM=${FILE_NAME%.*}; pipenv run python -u -m benchmarkfr ${FILE_NAME} --frame-rate 1 > ${FILE_STEM}.json 2> ${FILE_STEM}.log;' -- {}``

**Generating a timeline**

``pipenv run streamlit run timeline.py``
