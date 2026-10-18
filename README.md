# Hackathon-SwampXII
Built at SwampHacksXII (First Timers Track)

Live App: [Streamlit Link](https://hackathon-swampxii-j5j3u6hbuwptrjehde6ffm.streamlit.app/)

Grade Predictor that helps students see whether they are on track to pass a class, early enough to do something about it. Enter a few early-semester numbers, see the prediction, and get a personalized study plan from Gemini

**What It Does**
1. The student enters five inputs:
   - Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h per week)
   - Grades (G1), first-period grade, 0-20
   - Grades (G2), second-period grade, 0-20
   - Failures, past class failures, 0-3
   - Absences
   
2. A machine-learning model predicts whether the student is on track to pass (a final grade must be 10/20 or higher to count as passing).

3. The student is able to change the inputs to see how different habits change the outcome.

4. Pressing "Generate Gemini outline" sends the inputs and the prediction to the Gemini API, which returns a short personalized study outline.

**Tech Stack**
- Python, pandas, scikit-learn, joblib for the data and model.
- Google Colab for building and training the model.
- Streamlit (hosted on Streamlit Community Cloud) for the web app, with custom CSS and HTML for the design.
- Gemini API (Google) for the study outline.
- GitHub for version control and deployment.
- Google Fonts (Finger Paint, Press Start 2P).

**Project Structure**
- app. py                      #the Streamlit app
- requirements.txt             # Python packages Streamlit Cloud installs
- model.pkl                    # trained model
- .streamlit/confit.toml       # forces the light theme

**Use of AI Tools**
We used Claude (Anthropic) to help brainstorm and plan the project and to help write and debug parts of the code, including the Streamlit code and the Gemini integration. The Gemini API is part of the app itself.

**Limitations**
This is a 36-hour hackathon project. The prediction is an estimate based on a small dataset, not a guarantee, and it should be used as a prompt to act, not as a final judgment for any student.
