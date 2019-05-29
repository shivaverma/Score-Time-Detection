## Introduction:

- This Software detects the Score and Time written on the top left corner in the football match.
- For now Software supports only one particular format of the game.
- I am using a combination of LSTM and CNN networks to detect the information.
- Detction accuracy is more than 95%. 

### Input Image:

![](data/demo.png)

### Output detection

![](data/output.png)

##### Software language: Python 3.6
--------------------------------------------------------------------------------------------


`templates/` HTML page for the web page.

`requirements.txt` Required python packages.

`final_project.py` Main approach implementation.

`model/` Contain the pretrained model (LSTM + CNN).

`test.ipynb` Script to generate frames from a video.

`data/` It stores the image to detect, uploaded on the site.

`app.py` Python flask app. Uses the code from final_project.py

`jupyter.ipynb` Same implementation of final_project.py in ipython notebook.

--------------------------------------------------------------------------------------------

Football match format:          https://www.youtube.com/watch?v=1rWw2LkYzAQ&t=3403s

Reference Paper:                https://arxiv.org/pdf/1507.05717.pdf
